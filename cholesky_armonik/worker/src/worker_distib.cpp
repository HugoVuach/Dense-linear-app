// ============================================================================
//  Worker ArmoniK — Cholesky par blocs (DAG)
//  Opérations : POTRF, TRSM, SYRK, GEMM sur un bloc BxB
//  I/O via ResultsStore (IDs passés dans la payload : "blk/i/j")
//  Compute via Chameleon (Tile API) — backend StarPU/OpenMP selon build
// ============================================================================

// ============================================================================
//  Inclusions standard C++
// ============================================================================
#include <iostream>   // Flux d'entrée/sortie standard (std::cout, std::cerr, etc.)
#include <memory>     // Gestion mémoire intelligente (std::unique_ptr, std::shared_ptr)
#include <sstream>    // Flux de chaînes pour parser/formatter des données en texte
#include <string>     // Manipulation de chaînes de caractères std::string
#include <vector>     // Conteneur tableau dynamique 
#include <map>        // dictionnaire k=v
#include <stdexcept>  // Classes d'exceptions standard (std::runtime_error, etc.)
#include <cmath>      // Fonctions mathématiques #include <cstdlib>    // // getenv
#include <thread>     // Gestion des threads et requêtes sur le hardware 
#include <cstring>    // Memcpy
#include <cstdlib>    // Fonctions utilitaires standard 
#include <ctime>      // Fonctions liées au temps/calendrier 

// ============================================================================
//  Inclusions spécifiques gRPC
// ============================================================================

#include <grpcpp/grpcpp.h>                 // API principale gRPC pour C++
#include "grpcpp/support/sync_stream.h"    // Support pour les flux gRPC synchrones
#include "objects.pb.h"                    // Définition des messages gRPC 

// ============================================================================
//  Inclusions spécifiques à ArmoniK (SDK C++)
// ============================================================================

#include "utils/WorkerServer.h"             // Serveur Worker ArmoniK 
#include "Worker/ArmoniKWorker.h"           // Classe de base pour implémenter un Worker personnalisé
#include "Worker/ProcessStatus.h"           // Statut retourné après exécution d'une tâche
#include "Worker/TaskHandler.h"             // Objet représentant le contexte d'une tâche (payload, résultats attendus...)
#include "exceptions/ArmoniKApiException.h" // Gestion des exceptions spécifiques à l'API ArmoniK



// ============================================================================
//  Inclusions spécifiques à la bibliothèque Chameleon
// ============================================================================

// Chameleon est une bibliothèque HPC orientée calculs matriciels denses sur 
// architectures multi-core et hétérogènes (CPU + GPU).
// On utilise 'extern "C"' pour indiquer au compilateur C++ que les fonctions
// déclarées dans 'chameleon.h' suivent la convention de linkage du langage C
// afin d'éviter que le compilateur C++ applique le "name mangling",
// ce qui garantit que l'édition de liens (linker) trouve correctement les symboles
// définis dans la bibliothèque Chameleon compilée en C.

extern "C" { 
#include <chameleon.h>   
}


// ------------------------------ Utils ------------------------------

// ============================================================================
//  Fonction utilitaire : env_int
// ============================================================================
// Lecture d'une variable d'environnement et conversion en entier positif.
// Utile pour passer ngpu & ncpu en variables d'environnement
//
// Paramètres :
//  - key    : nom de la variable d'environnement à lire.
//  - defval : valeur par défaut à utiliser si la variable n'existe pas ou
//             si sa conversion échoue.
//
// Fonctionnement :
//  1) std::getenv(key) récupère la valeur associée à 'key' dans l'environnement.
//     - Si elle existe → renvoie un pointeur vers une chaîne C.
//     - Si elle n'existe pas → renvoie nullptr.
//  2) Si la variable existe, on tente de la convertir en entier avec std::stoi().
//     - std::stoi peut lever une exception si la conversion échoue (valeur non numérique).
//     - En cas de succès, on prend le maximum entre 0 et la valeur convertie
//       pour éviter les valeurs négatives.
//  3) Si la variable n'existe pas ou que la conversion échoue, on renvoie la
//     valeur par défaut 'defval'.

static int env_int(const char* key, int defval) {
  if (const char* s = std::getenv(key)) {      // 1) Lecture de la variable d'environnement
    try { 
      return std::max(0, std::stoi(s));        // 2) Conversion en entier positif
    } catch (...) {
      // Conversion échouée → on ignore et on utilisera defval
    }
  }
  return defval;                               // 3) Valeur par défaut
}

// ============================================================================
//  Fonction utilitaire : now_ts
// ============================================================================
// Retourne la date et l'heure actuelles sous forme d'une chaîne formatée.
//
// Fonctionnement :
// - std::time(nullptr) obtenir l'heure actuelle en secondes
// - std::localtime(&t) convertit cette valeur en une structure 'tm'
// - std::strftime(...) formate cette date dans le tampon 'buf' 

static std::string now_ts() {
  std::time_t t = std::time(nullptr);                          
  char buf[64];                                                
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S",         
                std::localtime(&t));                          
  return std::string(buf); 
}

// ============================================================================
//  Fonction utilitaire : parse_kv_payload
// ============================================================================
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

// ============================================================================
//  Fonction utilitaire : to_int
// ============================================================================
// Accès sûrs aux champs typés de la map k=v
static int to_int(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return std::stoi(it->second);
}

// ============================================================================
//  Fonction utilitaire : to_str
// ============================================================================
static std::string to_str(const std::map<std::string,std::string>& kv, const char* key) {
  auto it = kv.find(key);
  if (it == kv.end()) throw std::runtime_error(std::string("Missing key '") + key + "'");
  return it->second;
}

// ============================================================================
//  Fonction utilitaire : deserialize_block
// ============================================================================
// Désérialise un bloc BxB de doubles depuis un blob binaire (row-major)
static std::vector<double> deserialize_block(const std::string& blob, int B) {
  const size_t need = static_cast<size_t>(B) * static_cast<size_t>(B) * sizeof(double);
  if (blob.size() != need)
    throw std::runtime_error("Invalid block size: expected " + std::to_string(need) + " bytes, got " + std::to_string(blob.size()));
  std::vector<double> v(B*B);
  std::memcpy(v.data(), blob.data(), need);
  return v;
}

// ============================================================================
//  Fonction utilitaire : serialize_block
// ============================================================================
// Sérialise un bloc de doubles en blob binaire
static std::string serialize_block(const std::vector<double>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size()*sizeof(double));
}

// ============================================================================
//  Fonction utilitaire : download_blob
// ============================================================================
// I/O via TaskHandler : lecture/écriture d'un résultat ArmoniK (blob)
// - get_result(id).get() : télécharge le blob existant (entrée)
// - send_result(id, data).get() : publie la sortie sous l'ID attendu
static std::string download_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId) {
  return th.get_result(resultId).get();
}

// ============================================================================
//  Fonction utilitaire : upload_blob
// ============================================================================
static void upload_blob(armonik::api::worker::TaskHandler& th, const std::string& resultId, const std::string& data) {
  th.send_result(resultId, data).get();
}

// ============================================================================
//  Fonction utilitaire : create_desc_1block
// ============================================================================
// Construit un descripteur Chameleon pour un *seul* bloc BxB en place (1 tuile)
// - mb=nb=B ; bsiz=B*B ; grille 1x1 ; offsets 0.
static void create_desc_1block(CHAM_desc_t** desc, double* ptr, int B) {
  int mb=B, nb=B, bsiz=B*B, lm=B, ln=B, ioff=0, joff=0, m=B, n=B, p=1, q=1;
  CHAMELEON_Desc_Create(desc, (void*)ptr, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
}


















// ============================================================================
//  Classe : DAGCholeskyWorker
// ============================================================================
class DagCholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  explicit DagCholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent): ArmoniKWorker(std::move(agent)) {}

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
      // 2) Déterminer ncpu/ngpu via variables d'environnement
      // ----------------------------------------------------------------------
      int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency())); // // nb. threads CPU
      int ngpu = env_int("CHM_NGPU", 0);                                                       // // nb. GPUs visibles

      // ----------------------------------------------------------------------
      // Option : aligner sur le nombre de threads BLAS si précisé
      // int blas_threads = env_int("OPENBLAS_NUM_THREADS",
      //                    env_int("MKL_NUM_THREADS",
      //                    env_int("OMP_NUM_THREADS", ncpu)));
      //if (blas_threads > 0)
      //  ncpu = std::min(ncpu, blas_threads);
      // ----------------------------------------------------------------------

      std::cout << "[DagWorker] init CHAMELEON ncpu=" << ncpu << " ngpu=" << ngpu
                << " (BLAS_THREADS=" << blas_threads << ")\n";

      if (bsiz != mb * nb) {
        std::cout << "[CholeskyWorker] Warning: bsiz(" << bsiz << ") != mb*nb(" << (mb*nb) << ")\n";
      }

      // ----------------------------------------------------------------------
      // 3) Initialisation de Chameleon avec ncpu/ngpu
      // ----------------------------------------------------------------------                
      CHAMELEON_Init(ncpu, ngpu);

      int info = 0; // // code retour Chameleon (0 = OK)

      // ----------------------------------------------------------------------
      // 4) Préparer les buffers et descripteurs pour les blocs requis
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
      // 5) Exécuter le kernel Chameleon correspondant (Lower convention) 
      // + chronométrage/Glops
      // On mesure ici uniquement le temps du kernel Chameleon
      // Le calcul est différent pour chaque méthode 
      //     - POTRF(B) : 1/3 B*3
      //     - PTRSM(B×B, Right, triangular B) : 1/2 B*3
      //     - SYRK(B,B) : B*3
      //     - GEMM(B,B,B) : 2 B*3
      // ----------------------------------------------------------------------
      struct timespec t0{}, t1{};
      auto Bd = static_cast<double>(B);  // B en double
      double secs = 0.0;
      double flops = 0.0;                // nombre d'opérations flottantes
      double gflops = 0.0;               // GFLOP/s

      
      if (op == "POTRF") {
        // Factorisation de Cholesky (diag) : A := L * Lᵀ (stocké en Lower)
        clock_gettime(CLOCK_MONOTONIC, &t0);
        info = CHAMELEON_dpotrf_Tile(ChamLower, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        if (info != 0) throw std::runtime_error("dpotrf info=" + std::to_string(info));

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = (1.0/3.0) * Bd*Bd*Bd;         // ~ B^3 / 3
        gflops = flops / (secs * 1e9);

        std::cout << "[DagWorker] POTRF B=" << B
                  << " time=" << secs << " s"
                  << " perf=" << gflops << " GF/s\n";

        upload_blob(taskHandler, out, serialize_block(A)); // // publie L_kk
      }
      else if (op == "TRSM") {
        // "Descente" : Aik := Aik * inv(Lkkᵀ)  (Right,Lower,Trans,NonUnit)
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dtrsm_Tile(ChamRight, ChamLower, ChamTrans, ChamNonUnit, 1.0, dL, dA);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 0.5 * Bd*Bd*Bd;               // ~ B^3 / 2  (triangular solve par ligne)
        gflops = flops / (secs * 1e9);

        std::cout << "[DagWorker] TRSM  B=" << B
                  << " time=" << secs << " s"
                  << " perf=" << gflops << " GF/s\n";

        upload_blob(taskHandler, out, serialize_block(A)); // // publie L_ik
      }
      else if (op == "SYRK") {
        // Mise à jour diagonale : Cii := Cii - Aik * Aikᵀ
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dsyrk_Tile(ChamLower, ChamNoTrans, -1.0, dA, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = Bd*Bd*Bd;                     // ~ B^3 (demi-GEMM sur la partie triangulaire)
        gflops = flops / (secs * 1e9);

        std::cout << "[DagWorker] SYRK  B=" << B
                  << " time=" << secs << " s"
                  << " perf=" << gflops << " GF/s\n";

        upload_blob(taskHandler, out, serialize_block(C)); // // publie Cii maj
      }
      else { // GEMM
        // Mise à jour hors diagonale : Cij := Cij - Aik * Ajkᵀ
        clock_gettime(CLOCK_MONOTONIC, &t0);
        CHAMELEON_dgemm_Tile(ChamNoTrans, ChamTrans, -1.0, dAi, dAj, 1.0, dC);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        secs  = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        flops = 2.0 * Bd*Bd*Bd;               // 2 * B^3 (mult + add)
        gflops = flops / (secs * 1e9);

        std::cout << "[DagWorker] GEMM  B=" << B
                  << " time=" << secs << " s"
                  << " perf=" << gflops << " GF/s\n";        
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
