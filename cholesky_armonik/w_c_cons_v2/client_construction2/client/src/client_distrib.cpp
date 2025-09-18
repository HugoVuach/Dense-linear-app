
#include <string>                       
#include <sstream>                      
#include <vector>                      
#include <iostream> 
#include <map>                          
#include <random>                   
#include <algorithm>  
#include <unordered_map>
#include <cstring>
#include "objects.pb.h"
#include "utils/Configuration.h"
#include "logger/logger.h"             
#include "logger/writer.h"              
#include "logger/formatter.h"
#include "armonik/client/channel/ChannelFactory.h"    
#include "armonik/client/sessions/SessionsClient.h" 
#include "armonik/client/tasks/TasksClient.h"         
#include "armonik/client/results/ResultsClient.h"    
#include "armonik/client/events/EventsClient.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "absl/strings/string_view.h"
#include "cereal/archives/binary.hpp"
#include "cereal/types/vector.hpp"
#include <sstream>
#include <optional>
#include <stdexcept>
#include <cassert>
#include <cstdint>

namespace ak_common = armonik::api::common;
namespace ak_client = armonik::api::client;
namespace ak_grpc   = armonik::api::grpc::v1;

// =========================================================================================================================================================
// Etape : Parameterize N and B to avoid client Docker rebuilds
// =========================================================================================================================================================

struct Params {
  int N = 12; // default
  int B = 4; //default
};

static int parse_int_str(const std::string& s, int fallback, const char* name){
  try{
    size_t pos =0;
    long v = std::stol(s, &pos, 10);
    if (pos !=s.size()) throw std::invalid_argument("trailing chars");
    if (v <=0 || v > (1<<30)) throw std::out_of_range("range");
    return static_cast<int>(v);
  } catch(...){
    std::cerr << "[CONFIG] Ignoring invalid value for " << name << "='" << s << "', using " << fallback << "\n";
    return fallback;
  }
}
static Params load_params(int argc, char** argv) {
    Params p;
  // 1) Env vars
  if (const char* n = std::getenv("CHOLESKY_N")) p.N = parse_int_str(n, p.N, "CHOLESKY_N");
  if (const char* b = std::getenv("CHOLESKY_B")) p.B = parse_int_str(b, p.B, "CHOLESKY_B");

  // 2) CLI flags: --N=..., --B=..., et positionnels: N B
  auto eat = [](const std::string& arg, const char* prefix) -> std::optional<std::string> {
    const size_t len = std::strlen(prefix);
    if (arg.rfind(prefix, 0) == 0) return arg.substr(len);
    return std::nullopt;
  };

  int positional_seen = 0;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout <<
        "Usage: app [--N=INT] [--B=INT]\n"
        "   or: app N B\n"
        "Also supported via env: CHOLESKY_N / CHOLESKY_B\n";
      std::exit(0);
    }
    if (auto v = eat(arg, "--N=")) { p.N = parse_int_str(*v, p.N, "--N"); continue; }
    if (auto v = eat(arg, "--B=")) { p.B = parse_int_str(*v, p.B, "--B"); continue; }
    if (!arg.empty() && arg[0] != '-') {
      if (positional_seen == 0) { p.N = parse_int_str(arg, p.N, "N"); ++positional_seen; }
      else if (positional_seen == 1) { p.B = parse_int_str(arg, p.B, "B"); ++positional_seen; }
    }
  }

  if (p.N <= 0 || p.B <= 0) {
    throw std::invalid_argument("N and B must be positive");
  }
  return p;
}

/*
Entrées :
id_Cij = "5c60f693-bef5-e011-a485-80ee7300c695", id_Aik = "7f3a2c1d-8b9e-4a0b-9a7c-1e2d3c4b5a6f", id_Ajk = "9a7c1e2d-3c4b-5a6f-7f3a-2c1d8b9e4a0b", B = 448

Sortie (JSON) :
{"op":"GEMM","B":448,
"inC":"5c60f693-bef5-e011-a485-80ee7300c695",
"inAi":"7f3a2c1d-8b9e-4a0b-9a7c-1e2d3c4b5a6f",
"inAj":"9a7c1e2d-3c4b-5a6f-7f3a-2c1d8b9e4a0b"}
*/

/*
static std::string make_payload_potrf(const std::string& id_in, int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op"); w.String("POTRF");
  w.Key("B");  w.Int(B);
  w.Key("in"); w.String(id_in.c_str(), (rapidjson::SizeType)id_in.size());
  w.EndObject();
  return sb.GetString();
}

static std::string make_payload_trsm(const std::string& id_Lkk,
                                     const std::string& id_Aik,
                                     int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op");  w.String("TRSM");
  w.Key("B");   w.Int(B);
  w.Key("inL"); w.String(id_Lkk.c_str(), (rapidjson::SizeType)id_Lkk.size());
  w.Key("inA"); w.String(id_Aik.c_str(), (rapidjson::SizeType)id_Aik.size());
  w.EndObject();
  return sb.GetString();
}

static std::string make_payload_syrk(const std::string& id_Cii,
                                     const std::string& id_Aik,
                                     int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op");  w.String("SYRK");
  w.Key("B");   w.Int(B);
  w.Key("inC"); w.String(id_Cii.c_str(), (rapidjson::SizeType)id_Cii.size());
  w.Key("inA"); w.String(id_Aik.c_str(), (rapidjson::SizeType)id_Aik.size());
  w.EndObject();
  return sb.GetString();
}

static std::string make_payload_gemm(const std::string& id_Cij,
                                     const std::string& id_Aik,
                                     const std::string& id_Ajk,
                                     int B) {
  rapidjson::StringBuffer sb;
  rapidjson::Writer<rapidjson::StringBuffer> w(sb);
  w.StartObject();
  w.Key("op");   w.String("GEMM");
  w.Key("B");    w.Int(B);
  w.Key("inC");  w.String(id_Cij.c_str(), (rapidjson::SizeType)id_Cij.size());
  w.Key("inAi"); w.String(id_Aik.c_str(), (rapidjson::SizeType)id_Aik.size());
  w.Key("inAj"); w.String(id_Ajk.c_str(), (rapidjson::SizeType)id_Ajk.size());
  w.EndObject();
  return sb.GetString();
}
*/



static std::string make_payload(const std::vector<std::string>& ids, int B) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> w(sb);
    w.StartObject();
    w.Key("B"); w.Int(B);

    if (ids.size() == 1) {
        w.Key("op"); w.String("POTRF");
        w.Key("in"); w.String(ids[0].c_str(), (rapidjson::SizeType)ids[0].size());
    } else if (ids.size() == 2) {
        if (!ids[0].empty() && ids[0][0] == 'L') {
            w.Key("op"); w.String("TRSM");
            w.Key("inL"); w.String(ids[0].c_str(), (rapidjson::SizeType)ids[0].size());
            w.Key("inA"); w.String(ids[1].c_str(), (rapidjson::SizeType)ids[1].size());
        } else {
            w.Key("op"); w.String("SYRK");
            w.Key("inC"); w.String(ids[0].c_str(), (rapidjson::SizeType)ids[0].size());
            w.Key("inA"); w.String(ids[1].c_str(), (rapidjson::SizeType)ids[1].size());
        }
    } else if (ids.size() == 3) {
        w.Key("op"); w.String("GEMM");
        w.Key("inC");  w.String(ids[0].c_str(), (rapidjson::SizeType)ids[0].size());
        w.Key("inAi"); w.String(ids[1].c_str(), (rapidjson::SizeType)ids[1].size());
        w.Key("inAj"); w.String(ids[2].c_str(), (rapidjson::SizeType)ids[2].size());
    } else {
        throw std::runtime_error("Nombre d'arguments non supporté");
    }
    w.EndObject();
    return sb.GetString();
}






/*
static std::vector<double> generate_random_B_block(int B, double scale=1.0) 
{ 
static std::mt19937_64 rng(42); 
std::normal_distribution<double> dist(0.0, 1.0); 
std::vector<double> X(B*B); 
for (int t=0;t<B*B;++t) X[t] = dist(rng) * scale; return X; 
}
*/





// ------------------------------------------------------------------
// Etape : géneration de la matric SPD
// ------------------------------------------------------------------
/*
idx(i,j) = i + j*LDA → génères et symétrises en colonne-major : c’est exactement ce que CHAMELEON/LAPACK attend.
Tu imposes la symétrie en remplissant un triangle puis en recopiant l’autre : nickel.
Le bump diagonal + la fonction enforce_strict_diag_dominance garantissent une matrice SPD (via Gershgorin),
tant que tu les appliques après la symétrisation.
*/
static void make_spd_like_chameleon(double* A, int N, int LDA,
                                    double bump, char uplo, std::uint64_t seed)
{
    assert(A != nullptr);
    assert(N >= 0);
    assert(LDA >= std::max(1, N));

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    auto idx = [LDA](int i, int j) { return i + j * LDA; }; 

    if (uplo == 'L' || uplo == 'l') {
        for (int j = 0; j < N; ++j)
            for (int i = j; i < N; ++i)
                A[idx(i,j)] = dist(gen);
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < j; ++i)
                A[idx(i,j)] = A[idx(j,i)];
    } else { 
        for (int j = 0; j < N; ++j)
            for (int i = 0; i <= j; ++i)
                A[idx(i,j)] = dist(gen);
        for (int j = 0; j < N; ++j)
            for (int i = j+1; i < N; ++i)
                A[idx(i,j)] = A[idx(j,i)];
    }
    for (int i = 0; i < N; ++i) A[idx(i,i)] += bump;
}

// Option : forcer la SPD par stricte dominance diagonale
static void enforce_strict_diag_dominance(double* A, int N, int LDA, double eps = 1e-8)
{
    auto idx = [LDA](int i, int j) { return i + j * LDA; };
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int j = 0; j < N; ++j) if (j != i) s += std::abs(A[idx(i,j)]);
        double need = s + eps - A[idx(i,i)];
        if (need > 0.0) A[idx(i,i)] += need;
    }
}
// A en colonne-major, LDA >= N
// make_spd_like_chameleon(A, N, LDA, /*bump=*/100.0, 'L', /*seed=*/12345ULL);
// enforce_strict_diag_dominance(A, N, LDA);

// ------------------------------------------------------------------
// Etape : extraction des bloc en col-major
// ------------------------------------------------------------------
/*
extract_block_from_spd_matrix_colmajor(...) copie A[r0+ii, c0+jj] dans block[ii + jj*B] → col-major des deux côtés, parfait.
Zero-padding en bordure : OK.
Appel côté client avec A.data(), N, LDA (=N), B, bi, bj → c’est bon
*/
// Copie le bloc (bi,bj) de A (N×N, col-major, LDA) vers `block` (B×B, col-major).
// Zero-padding si le bloc dépasse les bords.
// A : pointeur sur la matrice globale en col-major (LDA >= N conseillé)
static void extract_block_from_spd_matrix_colmajor(const double* A,
                                                   int N, int LDA,
                                                   int B, int bi, int bj,
                                                   std::vector<double>& block)
{
  block.assign(static_cast<size_t>(B) * B, 0.0);

  // origine du bloc dans la matrice globale
  const int r0 = bi * B;  // row start
  const int c0 = bj * B;  // col start

  // indexeur col-major : a(i,j) = A[i + j*LDA]
  auto Aat = [A, LDA](int i, int j) -> double {
    return A[i + j * LDA];
  };

  // Remplir le buffer bloc (col-major aussi) : block[ii,jj] = A[r0+ii, c0+jj]
  for (int jj = 0; jj < B; ++jj) {
    int cj = c0 + jj;
    if (cj >= N) continue;                 // hors matrice → reste 0

    for (int ii = 0; ii < B; ++ii) {
      int ri = r0 + ii;
      if (ri >= N) break;                  // hors matrice → reste 0

      // block est col-major : index = ii + jj*B
      block[static_cast<size_t>(ii) + static_cast<size_t>(jj) * B] = Aat(ri, cj);
    }
  }
}






//  Prend en entrée deux entiers i et j et retourne une chaîne de caractères
// représentant l'identifiant du bloc au format "blk/i/j".
// ex : block_id(2, 1) → "blk/2/1"
static std::string block_id_from_ij(int i, int j) {
  std::ostringstream oss; oss << "blk/" << i << "/" << j; return oss.str();
}


//int main() {
int main(int argc, char** argv) {

  ak_common::logger::Logger logger{ak_common::logger::writer_console(),ak_common::logger::formatter_plain(true)};
  ak_common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  ak_client::ChannelFactory channelFactory(config, logger);
  std::shared_ptr<::grpc::Channel> channel = channelFactory.create_channel();
  ak_grpc::TaskOptions taskOptions;
  const std::string part_cpu_vm = "cholesky-cpu-vm";
  taskOptions.mutable_max_duration()->set_seconds(3600); 
  taskOptions.set_max_retries(3);                        
  taskOptions.set_priority(1);                           
  taskOptions.set_partition_id(part_cpu_vm);       
  taskOptions.set_application_name("cholesky-dag");         
  taskOptions.set_application_version("1.0");            
  taskOptions.set_application_namespace("benchmarks");
  ak_client::TasksClient    tasksClient(   ak_grpc::tasks::Tasks::NewStub(channel));
  ak_client::ResultsClient  resultsClient( ak_grpc::results::Results::NewStub(channel));
  ak_client::SessionsClient sessionsClient(ak_grpc::sessions::Sessions::NewStub(channel));
  ak_client::EventsClient   eventsClient(  ak_grpc::events::Events::NewStub(channel));

  // const int N  = 12;                   
  // const int B  = 4;
  const Params P = load_params(argc, argv);
  const int N = P.N;
  const int B = P.B;    
  const int Nb = (N + B - 1) / B;

  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu_vm});


// =========================================================================================================================================================
// Etape : création des ID blk/i/j et de leur ID associé
// =========================================================================================================================================================
  //                                                         ["blk/0/0", 
  // lower_triangle_block_keys.push_back(block_id(i,j))  ->   "blk/1/0", "blk/1/1",
  //                                                          "blk/2/0", "blk/2/1", "blk/2/2"]
  //
  // id_map →  {"blk/0/0": "15a98827-0053-47d8-86b8-9e0ef7250685",
  //                                [...]
  //            "blk/2/1": "db20d77b-524d-4f49-8e26-7551c54199b0"}

  std::vector<std::string> lower_triangle_block_keys; 
  lower_triangle_block_keys.reserve(Nb*(Nb+1)/2);
  for (int i=0;i<Nb;++i) 
    for (int j=0;j<=i;++j)                                          
      lower_triangle_block_keys.push_back(block_id_from_ij(i,j));   

  std::map<std::string,std::string> id_map = resultsClient.create_results_metadata(session_id, lower_triangle_block_keys); 
  std::cout << "[CLIENT][id_map initialisation] id_map size=" << id_map.size() << std::endl;
  for (const auto& [key, val] : id_map) {
    std::cout << "  " << key << " -> " << val << std::endl;
  }                                                                                                                        

// =========================================================================================================================================================
// Etape : création des blocs du triangle inférieur (diag comprise) + upload_result_data avec une ID initial propre à chaque block crée à l'étape precedente
// =========================================================================================================================================================

  // Pour chaque block std::vector<double> que l'on génère on les converti en absl::string_view  blockView
  // 
  //     * block
  //           - block = generate_random_B_block()
  //           - block.size() = 16 = 4 * 4 = B * B
  //
  //     * bytes
  //           - bytes = block.size() * sizeof(double) = 16 * 8 = 128
  //
  //     * blockView
  //           - conversion obligatoire pour respecter le format de upload_result()
  //           - blockView(reinterpret_cast<const char*>(block.data()),block.size()*sizeof(double))
  //             blockView(reinterpret_cast<const char*>(block.data()),128)
  //           - blockView.size() = 128 = block.size() * sizeof(double)


  //double ridge = 1e-6 * N;   
  //auto A = make_spd_from_gram(N, N, 1.0, ridge);
  // A allouée en col-major : std::vector<double> A(LDA * N);
  int LDA = N;
  std::vector<double> A((size_t)LDA * N);
  make_spd_like_chameleon(A.data(), N, LDA, /*bump=*/100.0, 'L', /*seed=*/12345ull);
  enforce_strict_diag_dominance(A.data(), N, LDA);

  for (int i=0;i<Nb;++i) {
    for (int j=0;j<=i;++j) {
      std::vector<double> block;
      extract_block_from_spd_matrix_colmajor(A.data(), N, LDA, B, i, j, block);
      absl::string_view blockView(reinterpret_cast<const char*>(block.data()),block.size() * sizeof(double));
      const std::string& result_id = id_map.at(block_id_from_ij(i,j));
      resultsClient.upload_result_data(session_id, result_id, blockView);


      /*
      // std::vector<double> block = generate_random_B_block(B, 0.1);
      // if (i==j) {
      //  for (int d=0; d<B; ++d) block[d*B + d] += (double)B;
      //}                                                                                                                
      auto bytes = block.size() * sizeof(double);                                                                      // Pour chaque block 
      std::cout << "[CLIENT][block = generate_random_B_block()] key=" << block_id_from_ij(i,j) << std::endl;           //            key = blk/i/j
      std::cout << "[CLIENT][block = generate_random_B_block()] block size =" << block.size() << std::endl;            //     block size = 16
      std::cout << "[CLIENT][block = generate_random_B_block()] byte =" << block.size()*sizeof(double) << std::endl;   //           byte = 128
      absl::string_view  blockView(reinterpret_cast<const char*>(block.data()),block.size()*sizeof(double));
      std::cout << "[CLIENT][blockView] blockView.size =" << blockView.size() << std::endl;                            // blockView.size = 128
      const std::string& result_id = id_map.at(block_id_from_ij(i,j));
      std::cout << "[CLIENT][result_id id_map] result_id =" << result_id << std::endl;                                 // c'est le bon result ID qui est associé à la clé blk/i/j
      resultsClient.upload_result_data(session_id, result_id, blockView);                                              // upload initial de chaque block
      */
    }
  }

// =========================================================================================================================================================
// Etape : summit_one
// =========================================================================================================================================================

//
// results["output"] → c’est l’ID de résultat où le worker doit écrire la sortie de la tâche (celui dans expected_output_keys).
// results["payload"] → c’est l’ID de résultat qui contient l’entrée de la tâche.

std::unordered_map<std::string, std::string> latest{id_map.begin(), id_map.end()};
std::cout << "[CLIENT][LATEST] latest size=" << latest.size() << std::endl;
for (const auto& [key, val] : latest) {
  std::cout << " [CLIENT][LATEST] latest key / value " << key << " -> " << val << std::endl;
}

// Soumet une tâche : payload_json = entrée, data_deps = IDs des résultats requis,
// partition_id = partition ArmoniK, output_id = rempli avec le nouvel ID de sortie créé.
// submit_one :
//        -  crée les deux résultats (payload pour l’entrée, output pour la sortie),
//        -  uploade le payload JSON,
//        -  soumet la tâche avec ses dépendances,
//        -  attend que la sortie soit disponible,
//        -  et retourne l’ID de sortie (output_id).
// 
// ajouter un download_result_data(session_id, Lkk_out) uniquement si pour inspecter le contenu

auto submit_one = [&](const std::string& payload_json,
                      std::vector<std::string> data_deps,
                      const std::string& partition_id ) {  
  
  std::cout << " [CLIENT][SUBMIT_ONE] : payload_json = " << payload_json << std::endl;                      // payload_json = {"op":"POTRF","B":4,"in":"15a98827-0053-47d8-86b8-9e0ef7250685"} qui a le même in que dans id_map
  std::cout << " [CLIENT][SUBMIT_ONE] : payload_json_len = " << payload_json.size() << std::endl;           // payload_json_len = 64
  std::cout << " [CLIENT][SUBMIT_ONE] : data_deps.size() = " << data_deps.size() << std::endl;              // data_deps.size() = 1 Normal pour POTRF
  std::cout << " [CLIENT][SUBMIT_ONE] : partition_id = " << partition_id << std::endl;                      // partition_id = cholesky-cpu-vm
                 
              
  // 1) Pour chaque tâche on recréer des Id de sortie car on ne peux pas réecrire le resultat sur le meme Id d'entrée : in 
  //  Crée deux résultats : un pour l'entrée ("payload"), un pour la sortie ("output")
  const auto result_ids = resultsClient.create_results_metadata(session_id, {"output", "payload"});         //
  const std::string output_id   = result_ids.at("output");
  const std::string& payload_id = result_ids.at("payload");
  std::cout <<  " [CLIENT][SUBMIT_ONE] : output_id = " << output_id << std::endl;                           //  output_id = 5d995457-3d16-4c16-be20-90deae7e02ae # c'est nouveau 
  std::cout <<  " [CLIENT][SUBMIT_ONE] : payload_id = " << payload_id << std::endl;                         // payload_id = ef13b5d9-2c3a-44ba-b026-5caeeb757485 # c'est nouveau

  // 2) Déduplique les dépendances ?
  std::sort(data_deps.begin(), data_deps.end());
  data_deps.erase(std::unique(data_deps.begin(), data_deps.end()), data_deps.end());
  std::cout << "[CLIENT][SUBMIT_ONE] : deps_ids=[";                                                         // deps_ids=[15a98827-0053-47d8-86b8-9e0ef7250685] qui est le meme que le in de payload_json
  for (size_t i = 0; i < data_deps.size(); ++i) {
    std::cout << (i ? "," : "") << data_deps[i];
  }
  std::cout << "]" << std::endl;

  resultsClient.upload_result_data(session_id, payload_id, payload_json);

  // 4) Soumission de la tâche : le worker écrira dans expected_output_keys[0]
  ak_common::TaskCreation tc;
  tc.payload_id           = payload_id;
  tc.expected_output_keys = { output_id };
  tc.data_dependencies    = std::move(data_deps);

  ak_grpc::TaskOptions per_task_opts = taskOptions;
  per_task_opts.set_partition_id(partition_id);

  std::cout << "[CLIENT][SUBMIT_ONE] : submit_tasks... expected_output=" << output_id << std::endl;
  tasksClient.submit_tasks(session_id, { tc }, per_task_opts);
  eventsClient.wait_for_result_availability(session_id, { output_id });

  std::cout << "[CLIENT][SUBMIT_ONE] : output available: " << output_id << std::endl;
  return output_id;
};

  // ------------------------ Exécution par vagues k -------------------------------
  for (int k = 0; k < Nb; ++k) {
    logger.info("Wave k=" + std::to_string(k));

    // 1) POTRF(k,k)
    {
      const std::string Lkk_in = latest.at(block_id_from_ij(k, k));                                                  // Lkk_in =15a98827-0053-47d8-86b8-9e0ef7250685 
      // const std::string payload = make_payload_potrf(Lkk_in, B); 
      const std::string payload = make_payload({Lkk_in}, B);
        std::cout << " [CLIENT][POTRF(k,k)] : blk" << block_id_from_ij(k, k) << std::endl;
        std::cout << " [CLIENT][POTRF(k,k)] :  Lkk_in =" << Lkk_in << std::endl;
        std::cout << " [CLIENT][POTRF(k,k)] : payload_len=" << payload.size() << std::endl;
        std::cout << " [CLIENT][POTRF(k,k)] :  payload_preview=" << payload.substr(0, 120)  << std::endl;

      const std::string Lkk_out = submit_one(payload, {Lkk_in}, part_cpu_vm);
      std::cout << " [CLIENT][POTRF(k,k)] : Lkk_out=" << Lkk_out  << std::endl;

      latest[block_id_from_ij(k, k)] = Lkk_out;
    }

    // 2) TRSM(i,k) pour i>k
    {
      const std::string Lkk_in = latest.at(block_id_from_ij(k, k));
      for (int i = k + 1; i < Nb; ++i) {
        const std::string Aik_in  = latest.at(block_id_from_ij(i, k));
        // const std::string payload = make_payload_trsm(Lkk_in, Aik_in, B);
        const std::string payload = make_payload({Lkk_in, Aik_in}, B);
        const std::string Aik_out = submit_one(payload, {Lkk_in, Aik_in}, part_cpu_vm);
        latest[block_id_from_ij(i, k)] = Aik_out;
      }
    }

    // 3) MAJ(i,j,k) : SYRK (diag) et GEMM (hors-diag)
    {
      const std::string part_for_update = part_cpu_vm; 

      for (int i = k + 1; i < Nb; ++i) {
        const std::string Aik_in = latest.at(block_id_from_ij(i, k));

        for (int j = k + 1; j <= i; ++j) {
          if (i == j) {
            const std::string Cii_in  = latest.at(block_id_from_ij(i, i));
            //const std::string payload = make_payload_syrk(Cii_in, Aik_in, B);
            const std::string payload = make_payload({Cii_in, Aik_in}, B);
            const std::string Cii_out = submit_one(payload, {Cii_in, Aik_in}, part_for_update);
            latest[block_id_from_ij(i, i)] = Cii_out;

          } else {
            const std::string Cij_in = latest.at(block_id_from_ij(i, j));
            const std::string Ajk_in = latest.at(block_id_from_ij(j, k));
            // const std::string payload = make_payload_gemm(Cij_in, Aik_in, Ajk_in, B);
            const std::string payload = make_payload({Cij_in, Aik_in, Ajk_in}, B); 
            const std::string Cij_out = submit_one(payload, {Cij_in, Aik_in, Ajk_in}, part_for_update);
            latest[block_id_from_ij(i, j)] = Cij_out;
          }
        }
      }
    }

    logger.info("Wave k=" + std::to_string(k) + " done.");
  }

  logger.info("All waves completed.");
  return 0;
}