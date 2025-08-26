// ============================================================================
//  Client ArmoniK — Cholesky par blocs (DAG distribué)
//  Vagues k : POTRF(k) → TRSM(i,k) pour i>k → MAJ(i,j,k) (SYRK/GEMM) pour k<j≤i
//  Chaque tâche lit/écrit des blocs via le store (IDs "blk/i/j").
// ============================================================================

// ============================================================================
//  Inclusions standard C++
// ============================================================================
#include <string>                       // Manipulation de chaînes de caractères std::string
#include <sstream>                      // Flux de chaînes pour parser/formatter des données en texte
#include <vector>                       // Conteneur tableau dynamique
#include <iostream>                     // Flux d'entrée/sortie standard

// ============================================================================
//  Inclusions spécifiques à ArmoniK & gRPC
// ============================================================================
#include "objects.pb.h"                 // Messages Protobuf (définitions gRPC)
#include "utils/Configuration.h"        // Gestion de la configuration (JSON, ENV)
#include "logger/logger.h"              // Logger ArmoniK (interface)
#include "logger/writer.h"              // Writer (console, fichier)
#include "logger/formatter.h"           // Format du logging (plain/json)
#include "channel/ChannelFactory.h"     // Usine à canaux gRPC selon la config
#include "sessions/SessionsClient.h"    // Client gRPC pour la gestion des sessions
#include "tasks/TasksClient.h"          // Client gRPC pour la soumission de tâches
#include "results/ResultsClient.h"      // Client gRPC pour la gestion des résultats
#include "events/EventsClient.h"        // Client gRPC pour s'abonner/attendre des événements

#include <map>                          //  std::map pour manipuler les IDs de résultats
#include <random>                       //  Génération de blocs aléatoires (POC)
#include <algorithm>                    //  Algorithmes STL (std::shuffle, std::generate)


// Alias de namespaces pour alléger l'écriture
namespace ak_common = armonik::api::common;
namespace ak_client = armonik::api::client;
namespace ak_grpc   = armonik::api::grpc::v1;

// ------------------------------ Utils ------------------------------

// ============================================================================
//  Fonction utilitaire : blk_id
// ============================================================================
// Construit un ID canonique pour un bloc (i,j) du triangle inférieur
// Convention : "blk/<i>/<j>"
static std::string blk_id(int i, int j) {
  std::ostringstream oss; oss << "blk/" << i << "/" << j; return oss.str();
}

// ============================================================================
//  Fonction utilitaire : make_payload_potrf
// ============================================================================
// Construit la payload d'une tâche POTRF sur le bloc diagonal (k,k)
// Format texte k=v séparés par espaces, facile à parser côté worker
static std::string make_payload_potrf(int k, int B) {
  std::ostringstream oss;
  oss << "op=POTRF k=" << k << " B=" << B
      << " in="  << blk_id(k,k)     // ID du bloc d'entrée (diag)
      << " out=" << blk_id(k,k);    // ID du bloc de sortie (écrasé)
  return oss.str();
}

// ============================================================================
//  Fonction utilitaire : make_payload_trsm
// ============================================================================
// Payload TRSM pour le bloc (i,k) en utilisant Lkk (k,k)
static std::string make_payload_trsm(int i, int k, int B) {
  std::ostringstream oss;
  oss << "op=TRSM i=" << i << " k=" << k << " B=" << B
      << " inL=" << blk_id(k,k)     // L_kk
      << " inA=" << blk_id(i,k)     // A_ik (sera transformé en L_ik)
      << " out=" << blk_id(i,k);    // sortie sur place
  return oss.str();
}

// ============================================================================
//  Fonction utilitaire : make_payload_syrk
// ============================================================================
// Payload SYRK pour la mise à jour du bloc diagonal (i,i) avec Aik
static std::string make_payload_syrk(int i, int k, int B) {
  std::ostringstream oss;
  oss << "op=SYRK i=" << i << " k=" << k << " B=" << B
      << " inC=" << blk_id(i,i)     // C_ii (sera mis à jour)
      << " inA=" << blk_id(i,k)     // A_ik
      << " out=" << blk_id(i,i);    // sortie sur place
  return oss.str();
}

// ============================================================================
//  Fonction utilitaire : make_payload_gemm
// ============================================================================
// Payload GEMM pour la mise à jour d'un bloc hors diagonale (i,j)
static std::string make_payload_gemm(int i, int j, int k, int B) {
  std::ostringstream oss;
  oss << "op=GEMM i=" << i << " j=" << j << " k=" << k << " B=" << B
      << " inC="  << blk_id(i,j)    // C_ij (sera mis à jour)
      << " inAi=" << blk_id(i,k)    // A_ik
      << " inAj=" << blk_id(j,k)    // A_jk
      << " out="  << blk_id(i,j);   // sortie sur place
  return oss.str();
}

// ============================================================================
//  Fonction utilitaire : gen_random_block
// ============================================================================
// Génère un bloc B×B aléatoire (POC) : utilisé pour injecter une matrice SPD-ish initiale
static std::vector<double> gen_random_block(int B, double scale=1.0) {
  static std::mt19937_64 rng(42);            // RNG fixe pour reproductibilité
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> X(B*B);
  for (int t=0;t<B*B;++t) X[t] = dist(rng) * scale;
  return X;
}

// ============================================================================
//  Fonction utilitaire : serialize_block
// ============================================================================
// Sérialise un bloc de doubles (row-major) en chaîne binaire (bytes) pour ResultsStore
static std::string serialize_block(const std::vector<double>& block) {
  return std::string(reinterpret_cast<const char*>(block.data()),
                     block.size()*sizeof(double));
}

// ------------------------------ Programme principal ------------------------------

int main() {
  // --------------------------------------------------------------------------
  // 1) Initialisation du logger et de la configuration
  // --------------------------------------------------------------------------
  ak_common::logger::Logger logger{ak_common::logger::writer_console(),ak_common::logger::formatter_plain(true)};
  ak_common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  logger.info("Initialized client config.");

  // --------------------------------------------------------------------------
  // 2) Création du canal gRPC vers ArmoniK 
  // --------------------------------------------------------------------------
  ak_client::ChannelFactory channelFactory(config, logger);
  std::shared_ptr<::grpc::Channel> channel = channelFactory.create_channel();

  // --------------------------------------------------------------------------
  // 3) Préparation des options de tâche (TaskOptions) et de la session
  // --------------------------------------------------------------------------
  ak_grpc::TaskOptions taskOptions;
  
  // Déclare deux partitions : CPU (par défaut) et GPU (si disponible)
  const std::string part_cpu = "cholesky-cpu";
  const std::string part_gpu = "cholesky-gpu"; // adapte au nom réel côté cluster
  const std::string default_partition = part_cpu;

  logger.info("Partitions (allowed in session): cpu=" + part_cpu + ", gpu=" + part_gpu);

  // Paramètres généraux de la tâche
  taskOptions.mutable_max_duration()->set_seconds(3600); // Durée max 
  taskOptions.set_max_retries(3);                        // 3 tentatives si échec
  taskOptions.set_priority(1);                           // Priorité
  taskOptions.set_partition_id(default_partition);       // Partition par défaut pour la session

  // Métadonnées
  taskOptions.set_application_name("cholesky-dag");      // Nom logique de l'application    
  taskOptions.set_application_version("1.0");            // Version
  taskOptions.set_application_namespace("benchmarks");

  // --------------------------------------------------------------------------
  // 4) Construction des clients ArmoniK (Sessions / Tasks / Results / Events)
  // --------------------------------------------------------------------------
  ak_client::TasksClient    tasksClient(   ak_grpc::tasks::Tasks::NewStub(channel));
  ak_client::ResultsClient  resultsClient( ak_grpc::results::Results::NewStub(channel));
  ak_client::SessionsClient sessionsClient(ak_grpc::sessions::Sessions::NewStub(channel));
  ak_client::EventsClient   eventsClient(  ak_grpc::events::Events::NewStub(channel));

  // --------------------------------------------------------------------------
  // 5) -------------------- Paramètres problème --------------------
  // -------------------------------------------------------------------------
  const int N  = 2048;                     // Taille globale de la matrice
  const int B  = 256;                      // Taille de bloc (tuile) carrée
  const int Nb = (N + B - 1) / B;          // Nombre de blocs par dimension (ici suppose N % B == 0)
  logger.info("Problem: N=" + std::to_string(N) + " B=" + std::to_string(B) + " Nb=" + std::to_string(Nb));

  // --------------------------------------------------------------------------
  // 6) Création d'une session d'exécution
  // --------------------------------------------------------------------------
  // On précise les partitions autorisées
  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu, part_gpu});
  logger.info("Session id = " + session_id);

  // Enregistrer les IDs de résultats pour tous les blocs (triangle inférieur)
  std::vector<std::string> all_block_names;
  all_block_names.reserve(Nb*Nb);
  for (int i=0;i<Nb;++i)
    for (int j=0;j<=i;++j)
      all_block_names.push_back(blk_id(i,j));
  std::map<std::string,std::string> id_map =
      resultsClient.create_results_metadata(session_id, all_block_names);


  // -------------------- Soumettre une tâche dans une partition donnée --------------------
  auto submit_partition = [&](const std::string& payload_name,
                                 const std::string& payload_text,
                                 const std::vector<std::string>& out_ids,
                                 const std::string& partition_id)
  {
    // 1) Créer et uploader le payload comme un Result
    auto meta = resultsClient.create_results_metadata(session_id, {payload_name});
    const std::string payload_id = meta[payload_name];
    resultsClient.upload_result_data(session_id, payload_id, payload_text);

    // 2) Construire la TaskCreation + TaskOptions spécifiques à la tâche
    ak_common::TaskCreation tc;
    tc.set_payload_id(payload_id);
    for (const auto& rid : out_ids)
      *tc.add_expected_output_keys() = rid;

    // Copier les options globales puis surcharger la partition pour CETTE tâche
    ak_grpc::TaskOptions per_task_opts = taskOptions;
    per_task_opts.set_partition_id(partition_id);
    *tc.mutable_task_options() = std::move(per_task_opts);

    // 3) Soumettre
    tasksClient.submit_tasks(session_id, { tc });
  };

  // -------------------- Injection des blocs initiaux --------------------
  // POC : on génère une matrice "SPD-ish" : blocs aléatoires, diagonale renforcée.
  logger.info("Uploading initial blocks (lower triangle)...");
  for (int i=0;i<Nb;++i) {
    for (int j=0;j<=i;++j) {
      std::vector<double> block = gen_random_block(B, 0.1);
      if (i==j) {
        // Renforce la diagonale pour rendre la matrice mieux conditionnée (positive définie)
        for (int d=0; d<B; ++d) block[d*B + d] += (double)B;
      }
      // Upload du bloc sérialisé dans le ResultsStore (clé -> id_map[clé])
      resultsClient.upload_result_data(session_id,
                                       id_map[blk_id(i,j)],
                                       serialize_block(block));
    }
  }
  logger.info("Initial blocks uploaded.");

  // -------------------- Exécution par vagues (k = 0..Nb-1) --------------------
  for (int k=0; k<Nb; ++k) {
    logger.info("Wave k=" + std::to_string(k));

    // ---- 1) POTRF(k,k) : factorise la diagonale (CPU) ----
    {
      std::string pl = make_payload_potrf(k, B);
      std::string payload_name = "payload/potrf/" + std::to_string(k);

      submit_partition(payload_name, pl, { id_map[blk_id(k,k)] }, part_cpu);
      // Synchronisation : on attend le bloc diagonal factorisé
      eventsClient.wait_for_result_availability(session_id, { id_map[blk_id(k,k)] });
    }

    // ---- 2) TRSM(i,k) pour i>k : descente de la colonne k (CPU) ----
    std::vector<std::string> trsm_out_ids;   // pour attendre en lot
    for (int i=k+1; i<Nb; ++i) {
      std::string pl = make_payload_trsm(i, k, B);
      std::string payload_name = "payload/trsm/" + std::to_string(i) + "/" + std::to_string(k);

      submit_partition(payload_name, pl, { id_map[blk_id(i,k)] }, part_cpu);
      trsm_out_ids.push_back(id_map[blk_id(i,k)]);
    }
    if (!trsm_out_ids.empty())
      eventsClient.wait_for_result_availability(session_id, trsm_out_ids);

    // ---- 3) MAJ(i,j,k) : SYRK (diag) et GEMM (off-diag)
    //      GPU si dispo, sinon CPU fallback ----
    const std::string part_for_update = part_gpu.empty() ? part_cpu : part_gpu;

    std::vector<std::string> upd_out_ids;
    for (int i=k+1; i<Nb; ++i) {
      for (int j=k+1; j<=i; ++j) {
        const bool is_diag = (i==j);
        std::string pl = is_diag ? make_payload_syrk(i, k, B)
                                 : make_payload_gemm(i, j, k, B);

        std::string payload_name = std::string("payload/") + (is_diag ? "syrk/" : "gemm/")
                                 + std::to_string(i) + "/" + std::to_string(j)
                                 + "/" + std::to_string(k);

        submit_partition(payload_name, pl, { id_map[blk_id(i,j)] }, part_for_update);
        upd_out_ids.push_back(id_map[blk_id(i,j)]);
      }
    }
    if (!upd_out_ids.empty())
      eventsClient.wait_for_result_availability(session_id, upd_out_ids);

    logger.info("Wave k=" + std::to_string(k) + " done.");
  }

  // -------------------- Fin --------------------
  logger.info("All waves completed. L factor is in blk/i/j (i≥j).");
  return 0;
}
