#include <Eigen/Dense>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdlib>
#include <utility>

#define NOMINMAX
#include <windows.h>

namespace fs = std::filesystem;
using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXd;
using Complex = std::complex<double>;

namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr double kDbFactor = 20.0 / 2.3025850929940456840179914546843642;
constexpr double kCEps = 1e-10;
constexpr double kRowEps = 1e-10;
constexpr double kLEps = 1e-8;
constexpr double kEdgeResistance = 1e-10;
constexpr uint64_t kStateMagic = 0x4652455154554e45ULL; // FREQTUNE

struct Options {
    fs::path root;
    fs::path model_dir;
    fs::path target_map;
    fs::path output_dir;
    double stop_mae_db = -1.0;
    int max_epochs = 120;
    double c_lr = 1e-2;
    double l_lr = 1e-3;
    std::string optimizer = "adam";
    std::string loss_kind = "huber";
    double huber_beta = 0.02;
    double s11_weight = 1.0;
    double s12_weight = 1.0;
    double prox_weight = 0.0;
    double grad_clip = 1.0;
    double weight_decay = 0.0;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double adam_eps = 1e-8;
    double momentum = 0.0;
    int checkpoint_every = 1;
    int scheduler_patience = 8;
    double scheduler_factor = 0.5;
    bool resume = false;
    bool check_gradient = false;
};

struct TargetRow {
    double frequency = 0.0;
    std::array<double, 8> values{};
};

struct Port {
    int positive = -1;
    int negative = -1;
    double z0 = 0.0;
};

struct Model {
    int n = 0;
    int e = 0;
    MatrixXd p0;
    MatrixXd l0;
    MatrixXd c0;
    MatrixXd ai;
    MatrixXd ap;
    std::array<Port, 2> ports;
    double c_scale = 0.0;
    double l_scale = 0.0;
    MatrixXd c_init_norm;
    MatrixXd l_init_norm;
    std::vector<double> initial_x;
    size_t c_count = 0;
    uint64_t input_hash = 0;
};

struct Built {
    MatrixXd c;
    MatrixXd l;
    MatrixXd c_norm;
    MatrixXd l_norm;
    MatrixXd factor;
};

struct Evaluation {
    double loss = 0.0;
    double prox = 0.0;
    double total_loss = 0.0;
    double combined_mae_db = 0.0;
    double s11_mae_db = 0.0;
    double s12_mae_db = 0.0;
    double rmse_db = 0.0;
    double max_abs_db = 0.0;
    std::vector<TargetRow> predicted;
    std::vector<double> gradient;
};

struct TrainingState {
    int epoch = 0;
    uint64_t step = 0;
    int best_epoch = 0;
    int bad_epochs = 0;
    double best_mae = std::numeric_limits<double>::infinity();
    double scheduler_best = std::numeric_limits<double>::infinity();
    double c_lr = 0.0;
    double l_lr = 0.0;
    std::vector<double> x;
    std::vector<double> m;
    std::vector<double> v;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string path_text(const fs::path& path) {
    return path.u8string();
}

double parse_double(const std::string& value, const std::string& name) {
    try {
        size_t end = 0;
        double result = std::stod(value, &end);
        if (end != value.size() || !std::isfinite(result)) fail("invalid " + name + ": " + value);
        return result;
    } catch (const std::exception&) {
        fail("invalid " + name + ": " + value);
    }
}

int parse_int(const std::string& value, const std::string& name) {
    try {
        size_t end = 0;
        int result = std::stoi(value, &end);
        if (end != value.size()) fail("invalid " + name + ": " + value);
        return result;
    } catch (const std::exception&) {
        fail("invalid " + name + ": " + value);
    }
}

void help() {
    std::cout
        << "Coarse_Freqtune.exe --stop-mae-db VALUE [options]\n\n"
        << "Default model: Coarse/Model\n"
        << "Default target: Coarse/Target_map.txt\n"
        << "Default output: Coarse/Freqtune_output\n"
        << "All target-map frequencies are used. Ports and reference impedances come only from Model/Port.txt.\n\n"
        << "Options:\n"
        << "  --stop-mae-db VALUE       Stop when mean(|S11 dB error|, |S12 dB error|) <= VALUE\n"
        << "  --max-epochs N            Maximum training epochs (default 120)\n"
        << "  --c-lr VALUE              C parameter learning rate (default 1e-2)\n"
        << "  --l-lr VALUE              L parameter learning rate (default 1e-3)\n"
        << "  --loss-kind KIND          huber (default), mae, or mse\n"
        << "  --huber-beta VALUE        Huber transition in dB (default 0.02)\n"
        << "  --s11-weight VALUE        Training-loss S11 weight (default 1)\n"
        << "  --s12-weight VALUE        Training-loss S12 weight (default 1)\n"
        << "  --optimizer KIND          adam (default), adamw, or sgd\n"
        << "  --weight-decay VALUE      Optimizer weight decay (default 0)\n"
        << "  --adam-beta1 VALUE        Adam first-moment factor (default 0.9)\n"
        << "  --adam-beta2 VALUE        Adam second-moment factor (default 0.999)\n"
        << "  --adam-eps VALUE          Adam denominator offset (default 1e-8)\n"
        << "  --sgd-momentum VALUE      SGD momentum (default 0)\n"
        << "  --prox-weight VALUE       Normalized C/L proximity penalty (default 0)\n"
        << "  --grad-clip VALUE         Global gradient norm cap (default 1)\n"
        << "  --checkpoint-every N      Write epoch snapshots every N epochs (default 1)\n"
        << "  --scheduler-patience N    LR plateau patience (default 8)\n"
        << "  --scheduler-factor VALUE  LR plateau factor (default 0.5)\n"
        << "  --model-dir PATH          Override default model directory\n"
        << "  --target-map PATH         Override default target map\n"
        << "  --output-dir PATH         Override default output directory\n"
        << "  --resume                  Continue from output/state.bin\n"
        << "  --check-gradient          Compare selected analytic gradients to finite differences\n"
        << "  --help                    Show this help\n";
}

fs::path executable_root() {
    std::wstring buffer(1024, L'\0');
    for (;;) {
        DWORD count = GetModuleFileNameW(nullptr, &buffer[0], static_cast<DWORD>(buffer.size()));
        if (count == 0) fail("could not resolve executable path");
        if (count < buffer.size()) {
            buffer.resize(count);
            return fs::path(buffer).parent_path();
        }
        if (buffer.size() >= 32768) fail("executable path is too long");
        buffer.resize(buffer.size() * 2);
    }
}

Options parse_options(int argc, char** argv) {
    Options o;
    o.root = executable_root();
    o.model_dir = o.root / "Coarse" / "Model";
    o.target_map = o.root / "Coarse" / "Target_map.txt";
    o.output_dir = o.root / "Coarse" / "Freqtune_output";
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") { help(); std::exit(0); }
        if (key == "--resume") { o.resume = true; continue; }
        if (key == "--check-gradient") { o.check_gradient = true; continue; }
        if (i + 1 >= argc) fail("missing value after " + key);
        std::string value = argv[++i];
        if (key == "--stop-mae-db") o.stop_mae_db = parse_double(value, key);
        else if (key == "--max-epochs") o.max_epochs = parse_int(value, key);
        else if (key == "--c-lr") o.c_lr = parse_double(value, key);
        else if (key == "--l-lr") o.l_lr = parse_double(value, key);
        else if (key == "--loss-kind") o.loss_kind = value;
        else if (key == "--huber-beta") o.huber_beta = parse_double(value, key);
        else if (key == "--s11-weight") o.s11_weight = parse_double(value, key);
        else if (key == "--s12-weight") o.s12_weight = parse_double(value, key);
        else if (key == "--optimizer") o.optimizer = value;
        else if (key == "--weight-decay") o.weight_decay = parse_double(value, key);
        else if (key == "--adam-beta1") o.beta1 = parse_double(value, key);
        else if (key == "--adam-beta2") o.beta2 = parse_double(value, key);
        else if (key == "--adam-eps") o.adam_eps = parse_double(value, key);
        else if (key == "--sgd-momentum") o.momentum = parse_double(value, key);
        else if (key == "--prox-weight") o.prox_weight = parse_double(value, key);
        else if (key == "--grad-clip") o.grad_clip = parse_double(value, key);
        else if (key == "--checkpoint-every") o.checkpoint_every = parse_int(value, key);
        else if (key == "--scheduler-patience") o.scheduler_patience = parse_int(value, key);
        else if (key == "--scheduler-factor") o.scheduler_factor = parse_double(value, key);
        else if (key == "--model-dir") o.model_dir = fs::absolute(fs::path(value));
        else if (key == "--target-map") o.target_map = fs::absolute(fs::path(value));
        else if (key == "--output-dir") o.output_dir = fs::absolute(fs::path(value));
        else fail("unknown option: " + key);
    }
    if (o.stop_mae_db < 0.0 && !o.check_gradient) fail("--stop-mae-db is required; use --help for options");
    if (o.max_epochs < 0 || o.checkpoint_every < 1 || o.scheduler_patience < 0) fail("invalid epoch/checkpoint/scheduler count");
    if (o.c_lr <= 0 || o.l_lr <= 0 || o.s11_weight <= 0 || o.s12_weight <= 0 ||
        o.huber_beta <= 0 || o.grad_clip <= 0 || o.prox_weight < 0 || o.weight_decay < 0 ||
        o.scheduler_factor <= 0 || o.scheduler_factor >= 1)
        fail("learning rates, weights, Huber beta and grad clip must be positive; scheduler factor must be in (0,1)");
    if (o.loss_kind != "huber" && o.loss_kind != "mae" && o.loss_kind != "mse") fail("invalid --loss-kind");
    if (o.optimizer != "adam" && o.optimizer != "adamw" && o.optimizer != "sgd") fail("invalid --optimizer");
    if (o.beta1 < 0 || o.beta1 >= 1 || o.beta2 < 0 || o.beta2 >= 1 ||
        o.adam_eps <= 0 || o.momentum < 0 || o.momentum >= 1)
        fail("invalid optimizer hyperparameters");
    return o;
}

std::ifstream open_input(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) fail("required input file missing or unreadable: " + path_text(path));
    return input;
}

MatrixXd read_header_matrix(const fs::path& path) {
    auto input = open_input(path);
    int rows = 0, cols = 0;
    if (!(input >> rows >> cols) || rows < 1 || cols < 1 || rows > 10000 || cols > 10000)
        fail("invalid matrix header: " + path_text(path));
    MatrixXd result(rows, cols);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        if (!(input >> result(i,j)) || !std::isfinite(result(i,j)))
            fail("matrix payload is incomplete or non-finite: " + path_text(path));
    }
    std::string extra;
    if (input >> extra) fail("matrix has surplus values: " + path_text(path));
    return result;
}

MatrixXd read_plain_matrix(const fs::path& path, int rows, int cols) {
    auto input = open_input(path);
    MatrixXd result(rows, cols);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        if (!(input >> result(i,j)) || !std::isfinite(result(i,j)))
            fail("incidence matrix has wrong dimensions or non-finite values: " + path_text(path));
    }
    std::string extra;
    if (input >> extra) fail("incidence matrix has surplus values: " + path_text(path));
    return result;
}

std::array<Port,2> read_ports(const fs::path& path, int n) {
    auto input = open_input(path);
    int count = 0;
    if (!(input >> count) || count != 2) fail("this version requires exactly two ports: " + path_text(path));
    std::array<Port,2> ports;
    for (auto& port : ports) {
        if (!(input >> port.positive >> port.negative >> port.z0) ||
            port.positive < 0 || port.positive >= n || port.negative < 0 || port.negative >= n ||
            port.positive == port.negative || !std::isfinite(port.z0) || port.z0 <= 0)
            fail("invalid port entry: " + path_text(path));
    }
    std::string extra;
    if (input >> extra) fail("Port.txt has surplus entries: " + path_text(path));
    return ports;
}

std::vector<TargetRow> read_target(const fs::path& path) {
    auto input = open_input(path);
    std::vector<TargetRow> rows;
    std::string line;
    int line_no = 0;
    while (std::getline(input, line)) {
        ++line_no;
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream stream(line);
        TargetRow row;
        if (!(stream >> row.frequency)) fail("invalid target frequency at line " + std::to_string(line_no));
        for (auto& value : row.values)
            if (!(stream >> value)) fail("target map must have nine numeric columns at line " + std::to_string(line_no));
        std::string extra;
        if (stream >> extra) fail("target map has surplus columns at line " + std::to_string(line_no));
        if (!std::isfinite(row.frequency) || row.frequency <= 0 ||
            (!rows.empty() && row.frequency <= rows.back().frequency))
            fail("target frequencies must be finite, positive and strictly increasing at line " + std::to_string(line_no));
        for (double value : row.values) if (!std::isfinite(value)) fail("non-finite target map value at line " + std::to_string(line_no));
        rows.push_back(row);
    }
    if (rows.empty()) fail("target map is empty: " + path_text(path));
    return rows;
}

uint64_t hash_files(const std::vector<fs::path>& paths) {
    uint64_t hash = 14695981039346656037ULL;
    for (const auto& path : paths) {
        auto input = open_input(path);
        char buffer[8192];
        while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0)
            for (std::streamsize i = 0; i < input.gcount(); ++i) {
                hash ^= static_cast<unsigned char>(buffer[i]);
                hash *= 1099511628211ULL;
            }
        hash ^= 0xff;
        hash *= 1099511628211ULL;
    }
    return hash;
}

double softplus(double x) {
    if (x > 30) return x + std::exp(-x);
    if (x < -30) return std::exp(x);
    return std::log1p(std::exp(x));
}

double softplus_inverse(double y) {
    y = std::max(y, 1e-12);
    return y + std::log(-std::expm1(-y));
}

double sigmoid(double x) {
    if (x >= 0) return 1.0 / (1.0 + std::exp(-x));
    double z = std::exp(x);
    return z / (1.0 + z);
}

Model load_model(const Options& o) {
    Model model;
    model.p0 = read_header_matrix(o.model_dir / "P.txt");
    model.l0 = read_header_matrix(o.model_dir / "L.txt");
    model.n = static_cast<int>(model.p0.rows());
    model.e = static_cast<int>(model.l0.rows());
    if (model.p0.cols() != model.n || model.l0.cols() != model.e) fail("P.txt and L.txt must be square matrices");
    if ((model.p0 - model.p0.transpose()).norm() > 1e-10 * model.p0.norm() ||
        (model.l0 - model.l0.transpose()).norm() > 1e-10 * model.l0.norm())
        fail("P.txt or L.txt is not symmetric");
    model.ports = read_ports(o.model_dir / "Port.txt", model.n);
    MatrixXd a = read_plain_matrix(o.model_dir / "A_E.txt", model.e + 2, model.n);
    MatrixXd at = read_plain_matrix(o.model_dir / "A_ET.txt", model.n, model.e + 2);
    if ((at + a.transpose()).norm() > 1e-10 * std::max(1.0, a.norm())) fail("A_ET.txt must equal -transpose(A_E.txt)");
    MatrixXd b2n = read_header_matrix(o.model_dir / "B2N.txt");
    if (b2n.rows() != model.e || b2n.cols() != 2) fail("B2N.txt must have one two-node row per branch");
    for (int row = 0; row < model.e; ++row) {
        int positive = static_cast<int>(std::llround(b2n(row,0)));
        int negative = static_cast<int>(std::llround(b2n(row,1)));
        if (positive < 0 || positive >= model.n || negative < 0 || negative >= model.n ||
            positive == negative || b2n(row,0) != positive || b2n(row,1) != negative)
            fail("invalid node pair in B2N.txt at branch " + std::to_string(row));
        Eigen::RowVectorXd expected = Eigen::RowVectorXd::Zero(model.n);
        expected(positive) = 1.0;
        expected(negative) = -1.0;
        if ((a.row(row) - expected).cwiseAbs().maxCoeff() > 1e-12)
            fail("B2N.txt and A_E.txt differ at branch " + std::to_string(row));
    }
    model.ai = a.topRows(model.e);
    model.ap.resize(2, model.n);
    std::array<int,2> matched{{-1,-1}};
    for (int p = 0; p < 2; ++p) {
        Eigen::RowVectorXd expected = Eigen::RowVectorXd::Zero(model.n);
        expected(model.ports[p].positive) = 1.0;
        expected(model.ports[p].negative) = -1.0;
        for (int row = 0; row < 2; ++row) {
            if ((a.row(model.e + row) - expected).cwiseAbs().maxCoeff() < 1e-12) {
                if (matched[p] != -1) fail("ambiguous port rows in A_E.txt");
                matched[p] = row;
            }
        }
        if (matched[p] < 0) fail("Port.txt entry does not match a port row of A_E.txt");
        model.ap.row(p) = a.row(model.e + matched[p]);
    }
    if (matched[0] == matched[1]) fail("both ports mapped to the same A_E row");
    Eigen::LDLT<MatrixXd> p_factor(model.p0);
    if (p_factor.info() != Eigen::Success || (p_factor.vectorD().array() <= 0).any()) fail("P.txt must be positive definite");
    model.c0 = p_factor.solve(MatrixXd::Identity(model.n, model.n));
    if (!model.c0.allFinite()) fail("could not invert P.txt");
    model.c_scale = model.c0.cwiseAbs().mean();
    model.l_scale = model.l0.cwiseAbs().mean();
    if (model.c_scale <= 0 || model.l_scale <= 0) fail("C or L has zero scale");
    MatrixXd cn = model.c0 / model.c_scale;
    MatrixXd ln = model.l0 / model.l_scale;
    Eigen::LLT<MatrixXd> chol(ln);
    if (chol.info() != Eigen::Success) fail("L.txt must be positive definite");
    MatrixXd factor = chol.matrixL();
    MatrixXd projected = MatrixXd::Zero(model.n, model.n);
    for (int i = 0; i < model.n; ++i) for (int j = 0; j < i; ++j) {
        double wij = std::max(-0.5 * (cn(i,j) + cn(j,i)), kCEps);
        model.initial_x.push_back(softplus_inverse(std::max(wij - kCEps, 1e-12)));
        projected(i,j) = projected(j,i) = -wij;
        projected(i,i) += wij;
        projected(j,j) += wij;
    }
    for (int i = 0; i < model.n; ++i) {
        double rowsum = std::max(cn(i,i) - projected(i,i), kRowEps);
        model.initial_x.push_back(softplus_inverse(std::max(rowsum - kRowEps, 1e-12)));
        projected(i,i) += rowsum;
    }
    model.c_count = model.initial_x.size();
    for (int i = 0; i < model.e; ++i) for (int j = 0; j < i; ++j) model.initial_x.push_back(factor(i,j));
    for (int i = 0; i < model.e; ++i)
        model.initial_x.push_back(softplus_inverse(std::max(factor(i,i) - kLEps, 1e-12)));
    model.c_init_norm = projected;
    model.l_init_norm = ln;
    model.input_hash = hash_files({o.model_dir / "P.txt", o.model_dir / "L.txt", o.model_dir / "A_E.txt",
                                   o.model_dir / "A_ET.txt", o.model_dir / "B2N.txt",
                                   o.model_dir / "Port.txt", o.target_map});
    std::cout << "model: nodes=" << model.n << " branches=" << model.e
              << " ports=2 parameters=" << model.initial_x.size()
              << " reference_ohms=" << model.ports[0].z0 << "," << model.ports[1].z0 << "\n";
    return model;
}

Built build_matrices(const Model& model, const std::vector<double>& x) {
    if (x.size() != model.initial_x.size()) fail("internal parameter count mismatch");
    Built b;
    b.c_norm = MatrixXd::Zero(model.n, model.n);
    b.factor = MatrixXd::Zero(model.e, model.e);
    size_t k = 0;
    for (int i = 0; i < model.n; ++i) for (int j = 0; j < i; ++j) {
        double w = softplus(x[k++]) + kCEps;
        b.c_norm(i,j) = b.c_norm(j,i) = -w;
        b.c_norm(i,i) += w;
        b.c_norm(j,j) += w;
    }
    for (int i = 0; i < model.n; ++i) b.c_norm(i,i) += softplus(x[k++]) + kRowEps;
    for (int i = 0; i < model.e; ++i) for (int j = 0; j < i; ++j) b.factor(i,j) = x[k++];
    for (int i = 0; i < model.e; ++i) b.factor(i,i) = softplus(x[k++]) + kLEps;
    b.l_norm = b.factor * b.factor.transpose();
    b.c = model.c_scale * b.c_norm;
    b.l = model.l_scale * b.l_norm;
    if (!b.c.allFinite() || !b.l.allFinite()) fail("non-finite trained matrix");
    return b;
}

double element_loss(double error, const Options& o, double* derivative) {
    double absolute = std::abs(error);
    if (o.loss_kind == "mse") { *derivative = 2.0 * error; return error * error; }
    if (o.loss_kind == "mae") { *derivative = (error > 0) - (error < 0); return absolute; }
    if (absolute < o.huber_beta) { *derivative = error / o.huber_beta; return 0.5 * error * error / o.huber_beta; }
    *derivative = (error > 0) - (error < 0);
    return absolute - 0.5 * o.huber_beta;
}

std::vector<double> parameter_gradient(const Model& model, const Built& b,
                                       const std::vector<double>& x, MatrixXd gc, MatrixXd gl,
                                       double prox_weight) {
    gc *= model.c_scale;
    gl *= model.l_scale;
    if (prox_weight > 0) {
        MatrixXd dc = b.c_norm - model.c_init_norm;
        MatrixXd dl = b.l_norm - model.l_init_norm;
        if (dc.norm() > 0) gc += (prox_weight / dc.norm()) * dc;
        if (dl.norm() > 0) gl += (prox_weight / dl.norm()) * dl;
    }
    MatrixXd gf = (gl + gl.transpose()) * b.factor;
    std::vector<double> g(x.size(), 0.0);
    size_t k = 0;
    for (int i = 0; i < model.n; ++i) for (int j = 0; j < i; ++j) {
        g[k] = sigmoid(x[k]) * (gc(i,i) + gc(j,j) - gc(i,j) - gc(j,i));
        ++k;
    }
    for (int i = 0; i < model.n; ++i) {
        g[k] = sigmoid(x[k]) * gc(i,i);
        ++k;
    }
    for (int i = 0; i < model.e; ++i) for (int j = 0; j < i; ++j) g[k++] = gf(i,j);
    for (int i = 0; i < model.e; ++i) {
        g[k] = sigmoid(x[k]) * gf(i,i);
        ++k;
    }
    return g;
}

Evaluation evaluate(const Model& model, const std::vector<TargetRow>& target,
                    const Options& o, const std::vector<double>& x, bool gradient,
                    bool retain_map = false) {
    Built b = build_matrices(model, x);
    Evaluation out;
    if (retain_map) out.predicted.reserve(target.size());
    MatrixXd gc = MatrixXd::Zero(model.n, model.n);
    MatrixXd gl = MatrixXd::Zero(model.e, model.e);
    MatrixXcd ai = model.ai.cast<Complex>();
    MatrixXcd ap = model.ap.cast<Complex>();
    Eigen::Matrix2cd inv_d = Eigen::Matrix2cd::Zero();
    inv_d(0,0) = 1.0 / std::sqrt(model.ports[0].z0);
    inv_d(1,1) = 1.0 / std::sqrt(model.ports[1].z0);
    double denominator = static_cast<double>(target.size()) * (o.s11_weight + o.s12_weight);
    double sum11 = 0, sum12 = 0, sum_sq = 0;
    for (const auto& row : target) {
        double omega = 2.0 * kPi * row.frequency;
        Complex jw(0.0, omega);
        MatrixXcd branch = jw * b.l.cast<Complex>();
        branch.diagonal().array() += Complex(kEdgeResistance, 0.0);
        Eigen::PartialPivLU<MatrixXcd> branch_lu(branch);
        MatrixXcd u = branch_lu.solve(ai);
        MatrixXcd nodal = jw * b.c.cast<Complex>() + ai.transpose() * u;
        Eigen::PartialPivLU<MatrixXcd> nodal_lu(nodal);
        MatrixXcd v = nodal_lu.solve(ap.transpose());
        if (!u.allFinite() || !v.allFinite()) fail("frequency solve failed at " + std::to_string(row.frequency) + " Hz");
        Eigen::Matrix2cd z = ap * v;
        Eigen::Matrix2cd zn = inv_d * z * inv_d;
        Eigen::Matrix2cd q = (zn + Eigen::Matrix2cd::Identity()).inverse();
        Eigen::Matrix2cd s = q * (zn - Eigen::Matrix2cd::Identity());
        if (!s.allFinite()) fail("non-finite S parameter at " + std::to_string(row.frequency) + " Hz");
        TargetRow pred;
        pred.frequency = row.frequency;
        int col = 0;
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
            Complex value = s(i,j);
            pred.values[col++] = kDbFactor * std::log(std::max(std::abs(value), std::numeric_limits<double>::min()));
            pred.values[col++] = std::arg(value) * 180.0 / kPi;
        }
        if (retain_map) out.predicted.push_back(pred);
        double err11 = pred.values[0] - row.values[0];
        double err12 = pred.values[2] - row.values[2];
        double d11 = 0, d12 = 0;
        out.loss += (o.s11_weight * element_loss(err11, o, &d11) +
                     o.s12_weight * element_loss(err12, o, &d12)) / denominator;
        sum11 += std::abs(err11);
        sum12 += std::abs(err12);
        sum_sq += err11 * err11 + err12 * err12;
        out.max_abs_db = std::max(out.max_abs_db, std::max(std::abs(err11), std::abs(err12)));
        if (!gradient) continue;
        Eigen::Matrix2cd sensitivity = Eigen::Matrix2cd::Zero();
        if (std::abs(s(0,0)) < 1e-250 || std::abs(s(0,1)) < 1e-250)
            fail("S11 or S12 is too close to zero for a stable dB gradient");
        sensitivity(0,0) = (o.s11_weight * d11 * kDbFactor / denominator) / s(0,0);
        sensitivity(0,1) = (o.s12_weight * d12 * kDbFactor / denominator) / s(0,1);
        Eigen::Matrix2cd gz_norm = q.transpose() * sensitivity * (Eigen::Matrix2cd::Identity() - s).transpose();
        Eigen::Matrix2cd gz = inv_d * gz_norm * inv_d;
        MatrixXcd h = branch_lu.solve(ai * v);
        MatrixXcd c_part = -jw * (v * gz * v.transpose());
        MatrixXcd l_part = jw * (h * gz * h.transpose());
        gc += c_part.real();
        gl += l_part.real();
    }
    out.s11_mae_db = sum11 / target.size();
    out.s12_mae_db = sum12 / target.size();
    out.combined_mae_db = 0.5 * (out.s11_mae_db + out.s12_mae_db);
    out.rmse_db = std::sqrt(sum_sq / (2.0 * target.size()));
    out.prox = o.prox_weight * ((b.c_norm - model.c_init_norm).norm() +
                                (b.l_norm - model.l_init_norm).norm());
    out.total_loss = out.loss + out.prox;
    if (gradient) out.gradient = parameter_gradient(model, b, x, gc, gl, o.prox_weight);
    if (!std::isfinite(out.total_loss) || !std::isfinite(out.combined_mae_db)) fail("non-finite training result");
    return out;
}

void write_matrix(const fs::path& path, const MatrixXd& matrix) {
    std::ofstream file(path);
    if (!file) fail("cannot write " + path_text(path));
    file << matrix.rows() << ' ' << matrix.cols() << '\n' << std::scientific << std::setprecision(18);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (j) file << ' ';
            file << matrix(i,j);
        }
        file << '\n';
    }
    if (!file) fail("write failed: " + path_text(path));
}

void write_map(const fs::path& path, const std::vector<TargetRow>& rows) {
    std::ofstream file(path);
    if (!file) fail("cannot write " + path_text(path));
    file << std::scientific << std::setprecision(10);
    for (const auto& row : rows) {
        file << row.frequency;
        for (double value : row.values) file << ',' << value;
        file << '\n';
    }
    if (!file) fail("write failed: " + path_text(path));
}

void export_result(const fs::path& dir, const Model& model, const Options& o,
                   const std::vector<double>& x, const Evaluation& ev, int epoch) {
    fs::create_directories(dir);
    Built b = build_matrices(model, x);
    Eigen::LDLT<MatrixXd> c_factor(b.c);
    if (c_factor.info() != Eigen::Success || (c_factor.vectorD().array() <= 0).any()) fail("trained C is not positive definite");
    MatrixXd p = c_factor.solve(MatrixXd::Identity(model.n, model.n));
    write_matrix(dir / "C.txt", b.c);
    write_matrix(dir / "L.txt", b.l);
    write_matrix(dir / "P.txt", p);
    for (const char* name : {"A_E.txt", "A_ET.txt", "Port.txt", "B2N.txt"}) {
        fs::path source = o.model_dir / name;
        if (fs::exists(source)) fs::copy_file(source, dir / name, fs::copy_options::overwrite_existing);
    }
    write_map(dir / "predicted_map.txt", ev.predicted);
    std::ofstream metrics(dir / "metrics.txt");
    metrics << std::setprecision(17)
            << "epoch " << epoch << '\n'
            << "combined_mae_db " << ev.combined_mae_db << '\n'
            << "s11_mae_db " << ev.s11_mae_db << '\n'
            << "s12_mae_db " << ev.s12_mae_db << '\n'
            << "rmse_db " << ev.rmse_db << '\n'
            << "max_abs_db " << ev.max_abs_db << '\n'
            << "training_loss " << ev.loss << '\n'
            << "prox_loss " << ev.prox << '\n'
            << "total_loss " << ev.total_loss << '\n';
}

template<class T> void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template<class T> void read_pod(std::istream& in, T& value) {
    if (!in.read(reinterpret_cast<char*>(&value), sizeof(value))) fail("truncated training state");
}
void write_vector(std::ostream& out, const std::vector<double>& values) {
    uint64_t count = values.size();
    write_pod(out, count);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(count * sizeof(double)));
}
void read_vector(std::istream& in, std::vector<double>& values, size_t expected) {
    uint64_t count = 0;
    read_pod(in, count);
    if (count != expected) fail("training state parameter count mismatch");
    values.resize(expected);
    if (!in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(double))))
        fail("truncated training state parameters");
}

void save_state(const fs::path& path, const TrainingState& state, uint64_t input_hash,
                const Options& o) {
    fs::path temp = path;
    temp += ".tmp";
    std::ofstream out(temp, std::ios::binary | std::ios::trunc);
    if (!out) fail("cannot write training state: " + path_text(temp));
    write_pod(out, kStateMagic);
    uint32_t version = 1;
    write_pod(out, version);
    write_pod(out, input_hash);
    write_pod(out, state.epoch);
    write_pod(out, state.step);
    write_pod(out, state.best_epoch);
    write_pod(out, state.bad_epochs);
    write_pod(out, state.best_mae);
    write_pod(out, state.scheduler_best);
    write_pod(out, state.c_lr);
    write_pod(out, state.l_lr);
    write_pod(out, o.huber_beta);
    write_pod(out, o.s11_weight);
    write_pod(out, o.s12_weight);
    write_pod(out, o.prox_weight);
    write_pod(out, o.beta1);
    write_pod(out, o.beta2);
    write_pod(out, o.adam_eps);
    write_pod(out, o.momentum);
    write_pod(out, o.weight_decay);
    write_pod(out, o.grad_clip);
    uint32_t loss_code = o.loss_kind == "huber" ? 0 : o.loss_kind == "mae" ? 1 : 2;
    uint32_t optimizer_code = o.optimizer == "adam" ? 0 : o.optimizer == "adamw" ? 1 : 2;
    write_pod(out, loss_code);
    write_pod(out, optimizer_code);
    write_vector(out, state.x);
    write_vector(out, state.m);
    write_vector(out, state.v);
    out.close();
    if (!out) fail("training state write failed: " + path_text(temp));
    std::error_code ec;
    fs::remove(path, ec);
    fs::rename(temp, path);
}

TrainingState load_state(const fs::path& path, uint64_t input_hash, size_t count, const Options& o) {
    auto in = open_input(path);
    uint64_t magic = 0, hash = 0;
    uint32_t version = 0, loss_code = 0, optimizer_code = 0;
    TrainingState state;
    read_pod(in, magic);
    read_pod(in, version);
    read_pod(in, hash);
    if (magic != kStateMagic || version != 1) fail("unsupported training state format");
    if (hash != input_hash) fail("training input files changed since state.bin was saved");
    read_pod(in, state.epoch);
    read_pod(in, state.step);
    read_pod(in, state.best_epoch);
    read_pod(in, state.bad_epochs);
    read_pod(in, state.best_mae);
    read_pod(in, state.scheduler_best);
    read_pod(in, state.c_lr);
    read_pod(in, state.l_lr);
    double beta, s11, s12, prox, beta1, beta2, adam_eps, momentum, weight_decay, grad_clip;
    read_pod(in, beta); read_pod(in, s11); read_pod(in, s12); read_pod(in, prox);
    read_pod(in, beta1); read_pod(in, beta2); read_pod(in, adam_eps);
    read_pod(in, momentum); read_pod(in, weight_decay); read_pod(in, grad_clip);
    read_pod(in, loss_code); read_pod(in, optimizer_code);
    uint32_t expected_loss = o.loss_kind == "huber" ? 0 : o.loss_kind == "mae" ? 1 : 2;
    uint32_t expected_optimizer = o.optimizer == "adam" ? 0 : o.optimizer == "adamw" ? 1 : 2;
    if (beta != o.huber_beta || s11 != o.s11_weight || s12 != o.s12_weight ||
        prox != o.prox_weight || beta1 != o.beta1 || beta2 != o.beta2 ||
        adam_eps != o.adam_eps || momentum != o.momentum ||
        weight_decay != o.weight_decay || grad_clip != o.grad_clip ||
        loss_code != expected_loss || optimizer_code != expected_optimizer)
        fail("training loss or optimizer options changed; cannot resume");
    read_vector(in, state.x, count);
    read_vector(in, state.m, count);
    read_vector(in, state.v, count);
    return state;
}

double clip_gradient(std::vector<double>& gradient, double cap) {
    double sum = 0;
    for (double value : gradient) {
        if (!std::isfinite(value)) fail("non-finite gradient");
        sum += value * value;
    }
    double norm = std::sqrt(sum);
    if (norm > cap) for (double& value : gradient) value *= cap / norm;
    return norm;
}

void optimizer_step(TrainingState& state, const Options& o, size_t c_count,
                    const std::vector<double>& gradient) {
    ++state.step;
    for (size_t i = 0; i < state.x.size(); ++i) {
        double lr = i < c_count ? state.c_lr : state.l_lr;
        double g = gradient[i];
        if (o.optimizer == "adamw") state.x[i] *= 1.0 - lr * o.weight_decay;
        else g += o.weight_decay * state.x[i];
        if (o.optimizer == "sgd") {
            state.m[i] = o.momentum * state.m[i] + g;
            state.x[i] -= lr * state.m[i];
        } else {
            state.m[i] = o.beta1 * state.m[i] + (1.0 - o.beta1) * g;
            state.v[i] = o.beta2 * state.v[i] + (1.0 - o.beta2) * g * g;
            double mhat = state.m[i] / (1.0 - std::pow(o.beta1, static_cast<double>(state.step)));
            double vhat = state.v[i] / (1.0 - std::pow(o.beta2, static_cast<double>(state.step)));
            state.x[i] -= lr * mhat / (std::sqrt(vhat) + o.adam_eps);
        }
        if (!std::isfinite(state.x[i])) fail("optimizer produced a non-finite parameter");
    }
}

void write_config(const Options& o, const Model& model, const std::vector<TargetRow>& target) {
    std::ofstream out(o.output_dir / "run_config.txt");
    if (!out) fail("cannot write run_config.txt");
    out << std::setprecision(17)
        << "model_dir " << path_text(o.model_dir) << '\n'
        << "target_map " << path_text(o.target_map) << '\n'
        << "output_dir " << path_text(o.output_dir) << '\n'
        << "input_hash_fnv1a64 " << model.input_hash << '\n'
        << "node_count " << model.n << '\n'
        << "branch_count " << model.e << '\n'
        << "port_1_z0_ohm " << model.ports[0].z0 << '\n'
        << "port_2_z0_ohm " << model.ports[1].z0 << '\n'
        << "frequency_rows " << target.size() << '\n'
        << "frequency_first_hz " << target.front().frequency << '\n'
        << "frequency_last_hz " << target.back().frequency << '\n'
        << "stop_mae_db " << o.stop_mae_db << '\n'
        << "max_epochs " << o.max_epochs << '\n'
        << "c_lr " << o.c_lr << '\n'
        << "l_lr " << o.l_lr << '\n'
        << "optimizer " << o.optimizer << '\n'
        << "loss_kind " << o.loss_kind << '\n'
        << "huber_beta " << o.huber_beta << '\n'
        << "s11_weight " << o.s11_weight << '\n'
        << "s12_weight " << o.s12_weight << '\n'
        << "prox_weight " << o.prox_weight << '\n'
        << "grad_clip " << o.grad_clip << '\n'
        << "checkpoint_every " << o.checkpoint_every << '\n'
        << "equation M=jwC+Ai^T(jwL+R)^-1Ai; Z=Ap M^-1 Ap^T; "
           "S=(D^-1 Z D^-1+I)^-1(D^-1 Z D^-1-I)\n";
}

void write_history_row(std::ofstream& out, int epoch, double pre_loss, const Evaluation& ev,
                       double grad_norm, const TrainingState& state, double seconds) {
    out << std::setprecision(16) << epoch << ',' << pre_loss << ',' << ev.loss << ','
        << ev.prox << ',' << ev.total_loss << ',' << ev.combined_mae_db << ','
        << ev.s11_mae_db << ',' << ev.s12_mae_db << ',' << ev.rmse_db << ','
        << ev.max_abs_db << ',' << grad_norm << ',' << state.c_lr << ','
        << state.l_lr << ',' << seconds << '\n';
    out.flush();
    if (!out) fail("could not write history.csv");
}

void write_summary(const Options& o, const TrainingState& state, const Evaluation& last,
                   bool reached) {
    std::ofstream out(o.output_dir / "summary.txt");
    if (!out) fail("cannot write summary.txt");
    out << std::setprecision(17)
        << "status " << (reached ? "target_reached" : "max_epochs_without_target") << '\n'
        << "stop_mae_db " << o.stop_mae_db << '\n'
        << "completed_epochs " << state.epoch << '\n'
        << "last_combined_mae_db " << last.combined_mae_db << '\n'
        << "best_epoch " << state.best_epoch << '\n'
        << "best_combined_mae_db " << state.best_mae << '\n'
        << "best_model_dir " << path_text(o.output_dir / "best") << '\n'
        << "history " << path_text(o.output_dir / "history.csv") << '\n';
}

void update_scheduler(TrainingState& state, const Options& o, double total_loss) {
    if (total_loss < state.scheduler_best * (1.0 - 1e-4)) {
        state.scheduler_best = total_loss;
        state.bad_epochs = 0;
    } else {
        ++state.bad_epochs;
        if (state.bad_epochs > o.scheduler_patience) {
            state.c_lr *= o.scheduler_factor;
            state.l_lr *= o.scheduler_factor;
            state.bad_epochs = 0;
        }
    }
}

void save_epoch_snapshot(const Options& o, const Model& model, const TrainingState& state,
                         const Evaluation& ev) {
    std::ostringstream name;
    name << "epoch_" << std::setw(4) << std::setfill('0') << state.epoch;
    export_result(o.output_dir / "checkpoints" / name.str(), model, o, state.x, ev, state.epoch);
}

void gradient_check(const Model& model, const std::vector<TargetRow>& target, const Options& o) {
    std::vector<TargetRow> selected{target.front()};
    Options local = o;
    local.prox_weight = 0;
    std::vector<double> x = model.initial_x;
    auto analytic = evaluate(model, selected, local, x, true);
    std::array<size_t, 8> indices{{0, model.c_count / 3, model.c_count - 1,
                                   model.c_count, model.c_count + model.e,
                                   x.size() - model.e - 1, x.size() - model.e, x.size() - 1}};
    double worst_relative = 0;
    for (size_t index : indices) {
        if (index >= x.size()) continue;
        double h = 1e-5 * std::max(1.0, std::abs(x[index]));
        x[index] += h;
        double plus = evaluate(model, selected, local, x, false).total_loss;
        x[index] -= 2*h;
        double minus = evaluate(model, selected, local, x, false).total_loss;
        x[index] += h;
        double numerical = (plus - minus) / (2*h);
        double relative = std::abs(numerical - analytic.gradient[index]) /
                          std::max({1e-7, std::abs(numerical), std::abs(analytic.gradient[index])});
        worst_relative = std::max(worst_relative, relative);
        std::cout << "gradient index=" << index << " analytic=" << analytic.gradient[index]
                  << " finite_difference=" << numerical << " relative=" << relative << '\n';
    }
    std::cout << "gradient check maximum relative error=" << worst_relative << '\n';
    if (worst_relative > 1e-2) fail("gradient check failed");
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options o = parse_options(argc, argv);
        auto target = read_target(o.target_map);
        Model model = load_model(o);
        std::cout << "target: rows=" << target.size() << " first_hz=" << target.front().frequency
                  << " last_hz=" << target.back().frequency << '\n';
        if (o.check_gradient) { gradient_check(model, target, o); return 0; }
        if (!o.resume && fs::exists(o.output_dir / "state.bin"))
            fail("output directory contains a previous run; use --resume or another --output-dir");
        fs::create_directories(o.output_dir);
        TrainingState state;
        state.x = model.initial_x;
        state.m.assign(state.x.size(), 0.0);
        state.v.assign(state.x.size(), 0.0);
        state.c_lr = o.c_lr;
        state.l_lr = o.l_lr;
        if (o.resume) {
            state = load_state(o.output_dir / "state.bin", model.input_hash, state.x.size(), o);
            std::cout << "resumed at epoch " << state.epoch << '\n';
        } else {
            write_config(o, model, target);
        }
        std::ofstream history(o.output_dir / "history.csv", o.resume ? std::ios::app : std::ios::trunc);
        if (!history) fail("cannot write history.csv");
        if (!o.resume)
            history << "epoch,pre_update_loss,post_update_loss,prox_loss,total_loss,combined_mae_db,"
                       "s11_mae_db,s12_mae_db,rmse_db,max_abs_db,grad_norm,c_lr,l_lr,train_seconds\n";
        Evaluation current = evaluate(model, target, o, state.x, false, true);
        if (!o.resume) {
            state.best_mae = current.combined_mae_db;
            state.scheduler_best = current.total_loss;
            export_result(o.output_dir / "best", model, o, state.x, current, 0);
            save_epoch_snapshot(o, model, state, current);
            write_history_row(history, 0, current.total_loss, current, 0.0, state, 0.0);
            save_state(o.output_dir / "state.bin", state, model.input_hash, o);
        }
        std::cout << std::setprecision(10) << "epoch " << state.epoch
                  << " combined_mae_db=" << current.combined_mae_db
                  << " stop_at=" << o.stop_mae_db << '\n';
        bool reached = current.combined_mae_db <= o.stop_mae_db;
        for (int epoch = state.epoch + 1; !reached && epoch <= o.max_epochs; ++epoch) {
            auto started = std::chrono::steady_clock::now();
            Evaluation before = evaluate(model, target, o, state.x, true);
            double grad_norm = clip_gradient(before.gradient, o.grad_clip);
            optimizer_step(state, o, model.c_count, before.gradient);
            Evaluation after = evaluate(model, target, o, state.x, false, true);
            state.epoch = epoch;
            if (after.combined_mae_db < state.best_mae) {
                state.best_mae = after.combined_mae_db;
                state.best_epoch = epoch;
                export_result(o.output_dir / "best", model, o, state.x, after, epoch);
            }
            reached = after.combined_mae_db <= o.stop_mae_db;
            if (epoch % o.checkpoint_every == 0 || reached || epoch == o.max_epochs)
                save_epoch_snapshot(o, model, state, after);
            update_scheduler(state, o, after.total_loss);
            save_state(o.output_dir / "state.bin", state, model.input_hash, o);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            write_history_row(history, epoch, before.total_loss, after, grad_norm, state, seconds);
            std::cout << "epoch " << epoch << " train_loss=" << after.total_loss
                      << " combined_mae_db=" << after.combined_mae_db
                      << " best=" << state.best_mae << " grad_norm=" << grad_norm
                      << " seconds=" << seconds << '\n';
            current = std::move(after);
        }
        write_summary(o, state, current, reached);
        std::cout << (reached ? "target reached" : "maximum epochs reached without target")
                  << "; best combined MAE=" << state.best_mae
                  << " dB at epoch " << state.best_epoch << '\n';
        return reached ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "Coarse_Freqtune error: " << error.what() << '\n';
        return 1;
    }
}
