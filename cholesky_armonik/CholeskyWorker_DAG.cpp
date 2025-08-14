// ============================================================================
//  Worker ArmoniK — Cholesky par blocs (DAG)
//  Opérations : POTRF, TRSM, SYRK, GEMM sur un bloc BxB
//  I/O via ResultsStore (IDs passés dans la payload : "blk/i/j")
//  Compute via Chameleon (Tile API) — backend StarPU/OpenMP selon build
// ============================================================================

#include <iostream>   // // logs console
#include <sstream>    // // parsing payload k=v
#include <string>     // // std::string
#include <vector>     // // buffers de blocs
#include <map>        // // dictionnaire k=v
#include <stdexcept>  // // exceptions
#include <cstdlib>    // // getenv
#include <thread>     // // hardware_concurrency
#include <cstring>    // // memcpy
#include <ctime>      // // horodatage

// ------------------------------ gRPC & ArmoniK SDK ------------------------------
#include <grpcpp/grpcpp.h>
#include "grpcpp/support/sync_stream.h"

#include "objects.pb.h"                 // // Protobuf générés (messages/services)
#include "utils/WorkerServer.h"         // // Serveur worker ArmoniK (boucle d'exécution)
#include "Worker/ArmoniKWorker.h"       // // Classe de base à dériver
#include "Worker/ProcessStatus.h"       // // Statut de fin d'exécution d'une tâche
#include "Worker/TaskHandler.h"         // // Contexte de tâche (payload, I/O résultats)
#include "exceptions/ArmoniKApiException.h" // // Exceptions spécifiques SDK

// ------------------------------ Chameleon (C) ------------------------------
extern "C" {
#include <chameleon.h>                  // // API Tile de Chameleon
}

// ------------------------------ Utils ------------------------------

// Récupère un entier depuis l'environnement (fallback si absent/illégal)
static int env_int(const char* key, int defval) {
  if (const char* s = std::getenv(key)) {
    try { return std::max(0, std::stoi(s)); } catch (...) {}
  }
  return defval;
}

// Horodatage lisible pour les logs
static std::string now_ts() {
  std::time_t t = std::time(nullptr);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
  return std::string(buf);
}

// Parse une payload "k=v k=v ..." en map<string,string>
static std::map<std::string,std::string> parse_kv_payload(const std::string& s) {
  std::map<std::string,std::string> kv;
  std::istringstream iss(s);
  std::string tok;
  while (iss >> tok) {
    auto pos = tok.find('=');
    if (pos != std::string::npos) kv[tok.substr(0,pos)] = tok.substr(pos+1);
  }
  return kv;
}

// Accès sûrs aux champs typés de la map k=v
static int to_int(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return std::stoi(it->second);
}
static std::string to_str(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return it->second;
}

// Désérialise un bloc BxB de doubles depuis un blob binaire (row-major)
static std::vector<double> deserialize_block(const std::string& blob, int B) {
  const size_t need = static_cast<size_t>(B) * static_cast<size_t>(B) * sizeof(double);
  if (blob.size() != need)
    throw std::runtime_error("Invalid block size: expected " + std::to_string(need) + " bytes, got " + std::to_string(blob.size()));
  std::vector<double> v(B*B);
  std::memcpy(v.data(), blob.data(), need);
  return v;
}

// Sérialise un bloc de doubles en blob binaire
static std::string serialize_block(const std::vector<double>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

// I/O via TaskHandler : lecture/écriture d'un résultat ArmoniK (blob)
// - get_result(id).get() : télécharge le blob existant (entrée)
// - send_result(id, data).get() : publie la sortie sous l'ID attendu
static std::string download_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId) {
  return th.get_result(resultId).get();
}
static void upload_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId, const std::string& data) {
  th.send_result(resultId, data).get();
}

// Construit un descripteur Chameleon pour un *seul* bloc BxB en place (1 tuile)
// - mb=nb=B ; bsiz=B*B ; grille 1x1 ; offsets 0.
static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
}

// ------------------------------ Worker ------------------------------

class DagCholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  // Le constructeur reçoit le stub Agent (gRPC) et le passe à la classe de base
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent)
      : ArmoniKWorker(std::move(agent)) {}

  // Execute() est appelée pour *chaque tâche* que le Compute Plane assigne à ce worker
  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    try {
      // ----------------------------------------------------------------------
      // 1) Récupérer et parser la payload k=v (opération + IDs de blocs + B)
      // ----------------------------------------------------------------------
      const std::string payload = taskHandler.getPayload();
      auto kv = parse_kv_payload(payload);

      const std::string op = to_str(kv, "op");   // // "POTRF" | "TRSM" | "SYRK" | "GEMM"
      const int B         = to_int(kv, "B");     // // taille de bloc (carré BxB)
      std::cout << "[DagWorker] " << now_ts() << " op=" << op << " B=" << B << "\n";

      // Indices (facultatifs) pour logs/debug (non utilisés dans ce kernel local)
      int i = kv.count("i") ? std::stoi(kv["i"]) : -1;
      int j = kv.count("j") ? std::stoi(kv["j"]) : -1;
      int k = kv.count("k") ? std::stoi(kv["k"]) : -1;
      (void)i; (void)j; (void)k;

      // IDs de résultats (entrées/sortie) selon l'opération
      //  - POTRF : in=blk/k/k → out=blk/k/k
      //  - TRSM  : inL=blk/k/k, inA=blk/i/k → out=blk/i/k
      //  - SYRK  : inC=blk/i/i, inA=blk/i/k → out=blk/i/i
      //  - GEMM  : inC=blk/i/j, inAi=blk/i/k, inAj=blk/j/k → out=blk/i/j
      std::string in, inL, inA, inC, inAi, inAj, out;
      if      (op == "POTRF") { in  = to_str(kv, "in");  out = to_str(kv, "out"); }
      else if (op == "TRSM")  { inL = to_str(kv,"inL");  inA = to_str(kv,"inA");  out= to_str(kv,"out"); }
      else if (op == "SYRK")  { inC = to_str(kv,"inC");  inA = to_str(kv,"inA");  out= to_str(kv,"out"); }
      else if (op == "GEMM")  { inC = to_str(kv,"inC");  inAi= to_str(kv,"inAi"); inAj= to_str(kv,"inAj"); out= to_str(kv,"out"); }
      else return armonik::api::worker::ProcessStatus("Unknown op=" + op);

      // ----------------------------------------------------------------------
      // 2) Initialiser Chameleon selon les ressources locales (ENV)
      // ----------------------------------------------------------------------
      int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency())); // // nb. threads CPU
      int ngpu = env_int("CHM_NGPU", 0);                                                       // // nb. GPUs visibles
      // Harmonise avec les threads BLAS si définis (évite l'over-subscription)
      int blas_threads = env_int("OPENBLAS_NUM_THREADS",
                           env_int("MKL_NUM_THREADS",
                           env_int("OMP_NUM_THREADS", ncpu)));
      if (blas_threads > 0) ncpu = std::min(ncpu, blas_threads);

      std::cout << "[DagWorker] init CHAMELEON ncpu=" << ncpu << " ngpu=" << ngpu
                << " (BLAS_THREADS=" << blas_threads << ")\n";
      CHAMELEON_Init(ncpu, ngpu);

      int info = 0; // // code retour Chameleon (0 = OK)

      // ----------------------------------------------------------------------
      // 3) Préparer les buffers et descripteurs pour les blocs requis
      // ----------------------------------------------------------------------
      std::vector<double> A(B*B), L(B*B), C(B*B), Ai(B*B), Aj(B*B);
      CHAM_desc_t *dA=nullptr, *dL=nullptr, *dC=nullptr, *dAi=nullptr, *dAj=nullptr;

      // Téléchargement des blocs d'entrée (depuis ResultsStore) + création des desc
      if (op == "POTRF") {
        A = deserialize_block(download_blob(taskHandler, in), B);
        create_desc_1block(&dA, A.data(), B);
      } else if (op == "TRSM") {
        L = deserialize_block(download_blob(taskHandler, inL), B);
        A = deserialize_block(download_blob(taskHandler, inA), B);
        create_desc_1block(&dL, L.data(), B);
        create_desc_1block(&dA, A.data(), B);
      } else if (op == "SYRK") {
        C = deserialize_block(download_blob(taskHandler, inC), B);
        A = deserialize_block(download_blob(taskHandler, inA), B);
        create_desc_1block(&dC, C.data(), B);
        create_desc_1block(&dA, A.data(), B);
      } else { // GEMM
        C  = deserialize_block(download_blob(taskHandler, inC),  B);
        Ai = deserialize_block(download_blob(taskHandler, inAi), B);
        Aj = deserialize_block(download_blob(taskHandler, inAj), B);
        create_desc_1block(&dC,  C.data(),  B);
        create_desc_1block(&dAi, Ai.data(), B);
        create_desc_1block(&dAj, Aj.data(), B);
      }

      // ----------------------------------------------------------------------
      // 4) Exécuter le kernel Chameleon correspondant (Lower convention)
      // ----------------------------------------------------------------------
      if (op == "POTRF") {
        // Factorisation de Cholesky (diag) : A := L * Lᵀ (stocké en Lower)
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        if (info != 0) throw std::runtime_error("dpotrf info=" + std::to_string(info));
        upload_blob(taskHandler, out, serialize_block(A)); // // publie L_kk
      }
      else if (op == "TRSM") {
        // "Descente" : Aik := Aik * inv(Lkkᵀ)  (Right,Lower,Trans,NonUnit)
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        upload_blob(taskHandler, out, serialize_block(A)); // // publie L_ik
      }
      else if (op == "SYRK") {
        // Mise à jour diagonale : Cii := Cii - Aik * Aikᵀ
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        upload_blob(taskHandler, out, serialize_block(C)); // // publie Cii maj
      }
      else { // GEMM
        // Mise à jour hors diagonale : Cij := Cij - Aik * Ajkᵀ
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        upload_blob(taskHandler, out, serialize_block(C)); // // publie Cij maj
      }

      // ----------------------------------------------------------------------
      // 5) Nettoyage des ressources Chameleon
      // ----------------------------------------------------------------------
      if (dA)  CHAMELEON_Desc_Destroy(&dA);
      if (dL)  CHAMELEON_Desc_Destroy(&dL);
      if (dC)  CHAMELEON_Desc_Destroy(&dC);
      if (dAi) CHAMELEON_Desc_Destroy(&dAi);
      if (dAj) CHAMELEON_Desc_Destroy(&dAj);
      CHAMELEON_Finalize(); // // termine la runtime (StarPU/OpenMP, etc.)

      // Succès
      return armonik::api::worker::ProcessStatus::Ok;
    }
    catch (const std::exception& e) {
      // En cas d'erreur, renvoie un statut explicite (visible côté Core/Client)
      std::cerr << "[DagWorker] ERROR: " << e.what() << "\n";
      return armonik::api::worker::ProcessStatus(e.what());
    }
  }
};

// ------------------------------ main() ------------------------------
// Démarre le serveur Worker ArmoniK avec ce DagCholeskyWorker
int main() {
  std::cout << "DagCholeskyWorker started. gRPC version = " << grpc::Version() << "\n";

  // Charge la configuration (fichier + ENV) et force les sockets Unix du Compute Plane
  armonik::api::common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  config.set("ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock"); // // socket Worker
  config.set("ComputePlane__AgentChannel__Address",  "/cache/armonik_agent.sock");  // // socket Agent

  try {
    // Crée un serveur Worker basé sur DagCholeskyWorker et lance la boucle d'exécution
    armonik::api::worker::WorkerServer::create<DagCholeskyWorker>(config)->run();
  } catch (const std::exception &e) {
    std::cerr << "Error in worker: " << e.what() << std::endl;
  }

  std::cout << "Stopping Server..." << std::endl;
  return 0;
}
