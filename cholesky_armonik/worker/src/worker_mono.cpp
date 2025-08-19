// ============================================================================
//  Inclusions standard C++
// ============================================================================

#include <iostream>   // Flux d'entrée/sortie standard (std::cout, std::cerr, etc.)
#include <memory>     // Gestion mémoire intelligente (std::unique_ptr, std::shared_ptr)
#include <sstream>    // Flux de chaînes pour parser/formatter des données en texte
#include <string>     // Manipulation de chaînes de caractères std::string
#include <vector>     // Conteneur tableau dynamique 
#include <stdexcept>  // Classes d'exceptions standard (std::runtime_error, etc.)
#include <cmath>      // Fonctions mathématiques 
#include <ctime>      // Fonctions liées au temps/calendrier 
#include <cstdlib>    // Fonctions utilitaires standard 
#include <thread>     // Gestion des threads et requêtes sur le hardware 

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
//  Classe : CholeskyWorker
// ============================================================================
// Hérite de ArmoniKWorker (classe de base du SDK ArmoniK C++) et implémente
// la méthode Execute() pour effectuer une factorisation de Cholesky en
// utilisant la bibliothèque Chameleon.
//
// La méthode Execute() est appelée par ArmoniK à chaque tâche reçue.
// Elle reçoit un TaskHandler qui fournit :
//   - la payload (paramètres d'entrée),
//   - la liste des résultats attendus,
//   - les méthodes pour publier les résultats.
//
// Ce worker est conçu pour lire les paramètres depuis la payload,
// récupérer la configuration CPU/GPU depuis les variables d'environnement,
// exécuter le calcul avec Chameleon, valider le résultat, et retourner
// un résumé texte en sortie.

class CholeskyWorker : public armonik::api::worker::ArmoniKWorker {
public:
  explicit CholeskyWorker(std::unique_ptr<armonik::api::grpc::v1::agent::Agent::Stub> agent): ArmoniKWorker(std::move(agent)) {}

  // Méthode exécutée pour chaque tâche
  armonik::api::worker::ProcessStatus Execute(armonik::api::worker::TaskHandler &taskHandler) override {
    try {
      // ----------------------------------------------------------------------
      // 1) Lecture et parsing de la payload
      // La payload attendue contient 14 entiers : N NB mb nb bsiz lm ln ioff joff m n p q seed
      // ----------------------------------------------------------------------
      const std::string payload = taskHandler.getPayload();
      std::istringstream iss(payload);
      int N, NB, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q, seed;
      
      // Si parsing échoue → statut d'erreur
      if (!(iss >> N >> NB >> mb >> nb >> bsiz >> lm >> ln >> ioff >> joff >> m >> n >> p >> q >> seed)) 
      {
        std::ostringstream err;
        err << "Payload parsing error. Expect 14 integers: "
              "N NB mb nb bsiz lm ln ioff joff m n p q seed. Got: " << payload;
        return armonik::api::worker::ProcessStatus(err.str());
      }

      // ----------------------------------------------------------------------
      // 2) Déterminer ncpu/ngpu via variables d'environnement
      // ----------------------------------------------------------------------
      int ncpu = env_int("CHM_NCPU", (int)std::max(1u, std::thread::hardware_concurrency()));
      int ngpu = env_int("CHM_NGPU", 0);

      // ----------------------------------------------------------------------
      // Option : aligner sur le nombre de threads BLAS si précisé
      // int blas_threads = env_int("OPENBLAS_NUM_THREADS",
      //                    env_int("MKL_NUM_THREADS",
      //                    env_int("OMP_NUM_THREADS", ncpu)));
      //if (blas_threads > 0)
      //  ncpu = std::min(ncpu, blas_threads);
      // ----------------------------------------------------------------------


      // Logs d'environnement et paramètres
      std::cout << "[CholeskyWorker] " << now_ts()
                << " env: CHM_NCPU=" << ncpu
                << " CHM_NGPU=" << ngpu
                << " (BLAS_THREADS=" << blas_threads << ")\n";

      std::cout << "[CholeskyWorker] payload: "
                << "N=" << N << " NB=" << NB
                << " mb=" << mb << " nb=" << nb << " bsiz=" << bsiz
                << " lm=" << lm << " ln=" << ln
                << " ioff=" << ioff << " joff=" << joff
                << " m=" << m << " n=" << n
                << " p=" << p << " q=" << q
                << " seed=" << seed << std::endl;

      if (bsiz != mb * nb) {
        std::cout << "[CholeskyWorker] Warning: bsiz(" << bsiz << ") != mb*nb(" << (mb*nb) << ")\n";
      }

      // ----------------------------------------------------------------------
      // 3) Initialisation de Chameleon avec ncpu/ngpu
      // ----------------------------------------------------------------------
      CHAMELEON_Init(ncpu, ngpu);

      int info = 0;
      CHAM_desc_t *descA = nullptr;
      CHAM_desc_t *descAorig = nullptr;
      CHAM_desc_t *descR = nullptr;

      // ----------------------------------------------------------------------
      // 4) Création et remplissage de la matrice SPD
      // ----------------------------------------------------------------------
      CHAMELEON_Desc_Create(&descA, NULL, ChamRealDouble,mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
      CHAMELEON_dplgsy_Tile((double)N, ChamLower, descA, seed);

      // Copie pour validation
      CHAMELEON_Desc_Create(&descAorig, NULL, ChamRealDouble,mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
      CHAMELEON_dlacpy_Tile(ChamUpperLower, descA, descAorig);

      // ----------------------------------------------------------------------
      // 5) Factorisation Cholesky et chronométrage
      // ----------------------------------------------------------------------
      struct timespec start{}, end{};
      clock_gettime(CLOCK_MONOTONIC, &start);
      info = CHAMELEON_dpotrf_Tile(ChamLower, descA);
      clock_gettime(CLOCK_MONOTONIC, &end);

      double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
      const double gflops = (1.0/3.0) * (double)N * (double)N * (double)N / (time_sec * 1e9);

      // ----------------------------------------------------------------------
      // 6) Validation numérique de la factorisation
      // ----------------------------------------------------------------------
      double normA = CHAMELEON_dlange_Tile(ChamInfNorm, descAorig);

      CHAMELEON_Desc_Create(&descR, NULL, ChamRealDouble, mb, nb, bsiz, lm, ln, ioff, joff, m, n, p, q);
      CHAMELEON_dlacpy_Tile(ChamLower, descA, descR);
      CHAMELEON_dlauum_Tile(ChamLower, descR); // R ← L * Lᵀ
      CHAMELEON_dgeadd_Tile(ChamNoTrans, -1.0, descR, 1.0, descAorig); // Aorig ← Aorig - LLᵀ

      double residual = CHAMELEON_dlange_Tile(ChamInfNorm, descAorig);
      double relative_error = residual / (normA > 0.0 ? normA : 1.0);

      // ----------------------------------------------------------------------
      // 7) Libération des ressources Chameleon
      // ----------------------------------------------------------------------
      CHAMELEON_Desc_Destroy(&descA);
      CHAMELEON_Desc_Destroy(&descAorig);
      CHAMELEON_Desc_Destroy(&descR);
      CHAMELEON_Finalize();

      // ----------------------------------------------------------------------
      // 8) Préparer un résumé texte du calcul
      // ----------------------------------------------------------------------
      std::ostringstream out;
      out.setf(std::ios::fixed); out.precision(3);
      out << "Cholesky with Chameleon (LL^T)\n"
          << "N=" << N << ", NB=" << NB << "\n"
          << "Time=" << time_sec << " s\n"
          << "Perf=" << gflops << " Gflop/s\n";

      out.setf(std::ios::scientific); out.precision(3);
      out << "RelError=" << relative_error << "\n"
          << "Status=" << (info == 0 ? "OK" : ("CHAMELEON_dpotrf_Tile info=" + std::to_string(info))) << "\n";

      std::string result_str = out.str();
      std::cout << "[CholeskyWorker] Result summary:\n" << result_str << std::endl;

      // ----------------------------------------------------------------------
      // 9) Publier le résultat si attendu par le client
      // ----------------------------------------------------------------------
      try {
        const auto &expected = taskHandler.getExpectedResults();
        if (!expected.empty()) {
          taskHandler.send_result(expected[0], result_str).get();
        }
      } catch (const std::exception &e) {
        std::cerr << "[CholeskyWorker] Error sending result: " << e.what() << std::endl;
        return armonik::api::worker::ProcessStatus(e.what());
      }

      // ----------------------------------------------------------------------
      // 10) Retourner le statut d'exécution
      // ----------------------------------------------------------------------
      if (info != 0) 
      {
        return armonik::api::worker::ProcessStatus("CHAMELEON_dpotrf_Tile failed with info=" + std::to_string(info));
      }
      
      return armonik::api::worker::ProcessStatus::Ok;

    } catch (const std::exception &e) {
      // Gestion d'exception générale
      std::cerr << "[CholeskyWorker] Fatal error: " << e.what() << std::endl;
      return armonik::api::worker::ProcessStatus(e.what());
    }
  }
};


// ============================================================================
// Ce main initialise la configuration ArmoniK pour le worker
// ============================================================================
int main() {
  // Initialisation
  std::cout << "CholeskyWorker started. gRPC version = " << grpc::Version() << "\n";

  // Prépare et charge la configuration + ajoute les variables d'environnement.
  armonik::api::common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();

  // Définit explicitement l'adresse des sockets Unix pour communiquer :
  config.set("ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock");
  config.set("ComputePlane__AgentChannel__Address", "/cache/armonik_agent.sock");

  // Crée et lance un serveur Worker qui utilisera CholeskyWorker pour chaque tâche reçue.
  try {
    armonik::api::worker::WorkerServer::create<CholeskyWorker>(config)->run();

  // Capture toute exception survenue 
  } catch (const std::exception &e) {
    std::cerr << "Error in worker: " << e.what() << std::endl;
  }

  // Fermeture
  std::cout << "Stopping Server..." << std::endl;
  return 0;
}
