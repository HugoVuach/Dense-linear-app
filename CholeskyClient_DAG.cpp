// ============================================================================
//  Client ArmoniK — Cholesky par blocs (DAG distribué)
//  Vagues k : POTRF(k) → TRSM(i,k) pour i>k → MAJ(i,j,k) (SYRK/GEMM) pour k<j≤i
//  Chaque tâche lit/écrit des blocs via le store (IDs "blk/i/j").
// ============================================================================

#include "objects.pb.h"                 // // Définitions Protobuf/gRPC générées (messages, services)
#include "utils/Configuration.h"        // // Helper pour charger config depuis JSON + variables d'environnement
#include "logger/logger.h"              // // Logger ArmoniK (API)
#include "logger/writer.h"              // // Sortie des logs (console, fichier, etc.)
#include "logger/formatter.h"           // // Format des logs (plain text / json)
#include "channel/ChannelFactory.h"     // // Fabrique pour créer le canal gRPC selon la configuration
#include "sessions/SessionsClient.h"    // // Client gRPC pour créer/fermer les sessions
#include "tasks/TasksClient.h"          // // Client gRPC pour soumettre des tâches
#include "results/ResultsClient.h"      // // Client gRPC pour créer/charger/sauver des résultats (blobs)
#include "events/EventsClient.h"        // // Client gRPC pour attendre des événements (ex: résultat disponible)

#include <map>                          // // std::map : dictionnaire nom→id de résultat
#include <string>                       // // std::string
#include <sstream>                      // // std::ostringstream pour construire des payloads texte
#include <vector>                       // // std::vector : listes de noms/ids, buffers, etc.
#include <random>                       // // Génération de blocs aléatoires (POC)
#include <iostream>                     // // (optionnel) sorties console

// Alias de namespaces pour alléger l'écriture
namespace ak_common = armonik::api::common;
namespace ak_client = armonik::api::client;
namespace ak_grpc   = armonik::api::grpc::v1;

// ------------------------------ Utilitaires ------------------------------

// Construit un ID canonique pour un bloc (i,j) du triangle inférieur
// Convention : "blk/<i>/<j>"
static std::string blk_id(int i, int j) {
  std::ostringstream oss; oss << "blk/" << i << "/" << j; return oss.str();
}

// Construit la payload d'une tâche POTRF sur le bloc diagonal (k,k)
// Format texte k=v séparés par espaces, facile à parser côté worker
static std::string make_payload_potrf(int k, int B) {
  std::ostringstream oss;
  oss << "op=POTRF k=" << k << " B=" << B
      << " in="  << blk_id(k,k)     // // ID du bloc d'entrée (diag)
      << " out=" << blk_id(k,k);    // // ID du bloc de sortie (écrasé)
  return oss.str();
}

// Payload TRSM pour le bloc (i,k) en utilisant Lkk (k,k)
static std::string make_payload_trsm(int i, int k, int B) {
  std::ostringstream oss;
  oss << "op=TRSM i=" << i << " k=" << k << " B=" << B
      << " inL=" << blk_id(k,k)     // // L_kk
      << " inA=" << blk_id(i,k)     // // A_ik (sera transformé en L_ik)
      << " out=" << blk_id(i,k);    // // sortie sur place
  return oss.str();
}

// Payload SYRK pour la mise à jour du bloc diagonal (i,i) avec Aik
static std::string make_payload_syrk(int i, int k, int B) {
  std::ostringstream oss;
  oss << "op=SYRK i=" << i << " k=" << k << " B=" << B
      << " inC=" << blk_id(i,i)     // // C_ii (sera mis à jour)
      << " inA=" << blk_id(i,k)     // // A_ik
      << " out=" << blk_id(i,i);    // // sortie sur place
  return oss.str();
}

// Payload GEMM pour la mise à jour d'un bloc hors diagonale (i,j)
static std::string make_payload_gemm(int i, int j, int k, int B) {
  std::ostringstream oss;
  oss << "op=GEMM i=" << i << " j=" << j << " k=" << k << " B=" << B
      << " inC="  << blk_id(i,j)    // // C_ij (sera mis à jour)
      << " inAi=" << blk_id(i,k)    // // A_ik
      << " inAj=" << blk_id(j,k)    // // A_jk
      << " out="  << blk_id(i,j);   // // sortie sur place
  return oss.str();
}

// Génère un bloc B×B aléatoire (POC) : utilisé pour injecter une matrice SPD-ish initiale
static std::vector<double> gen_random_block(int B, double scale=1.0) {
  static std::mt19937_64 rng(42);            // // RNG fixe pour reproductibilité
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> X(B*B);
  for (int t=0;t<B*B;++t) X[t] = dist(rng) * scale;
  return X;
}

// Sérialise un bloc de doubles (row-major) en chaîne binaire (bytes) pour ResultsStore
static std::string serialize_block(const std::vector<double>& block) {
  return std::string(reinterpret_cast<const char*>(block.data()),
                     block.size()*sizeof(double));
}

// ------------------------------ Programme principal ------------------------------

int main() {
  // -------------------- Logger + Configuration --------------------
  ak_common::logger::Logger logger{
    ak_common::logger::writer_console(),         // // Log sur la console
    ak_common::logger::formatter_plain(true)     // // Format simple avec horodatage
  };
  ak_common::utils::Configuration config;
  config.add_json_configuration("/appsettings.json") // // Charge /appsettings.json si présent
        .add_env_configuration();                    // // Écrase/complète avec variables d'environnement
  logger.info("Initialized client config.");

  // Crée le canal gRPC vers ArmoniK à partir de la config (adresse, TLS...).
  ak_client::ChannelFactory channelFactory(config, logger);
  std::shared_ptr<::grpc::Channel> channel = channelFactory.create_channel();

  // -------------------- Options de session / tâches --------------------
  ak_grpc::TaskOptions taskOptions;
  const std::string part_cpu = "cholesky-cpu";   // // Partition cible par défaut (à adapter)
  const std::string part_gpu = "cholesky-cpu";   // // Exemple si tu as une partition GPU dédiée
  logger.info("Partitions: cpu=" + part_cpu + ", gpu=" + part_gpu);

  taskOptions.mutable_max_duration()->set_seconds(3600);  // // Timeout max par tâche (1h)
  taskOptions.set_max_retries(3);                         // // Retries en cas d'échec
  taskOptions.set_priority(1);                            // // Priorité relative
  taskOptions.set_partition_id(part_cpu);                 // // Partition par défaut pour la session

  // Métadonnées applicatives (pour traçabilité/monitoring)
  taskOptions.set_application_name("cholesky-dag");
  taskOptions.set_application_version("1.0");
  taskOptions.set_application_namespace("benchmarks");

  // -------------------- Création des clients gRPC ArmoniK --------------------
  ak_client::TasksClient    tasksClient(   ak_grpc::tasks::Tasks::NewStub(channel));
  ak_client::ResultsClient  resultsClient( ak_grpc::results::Results::NewStub(channel));
  ak_client::SessionsClient sessionsClient(ak_grpc::sessions::Sessions::NewStub(channel));
  ak_client::EventsClient   eventsClient(  ak_grpc::events::Events::NewStub(channel));

  // -------------------- Paramètres problème --------------------
  const int N  = 2048;                     // // Taille globale de la matrice
  const int B  = 256;                      // // Taille de bloc (tuile) carrée
  const int Nb = (N + B - 1) / B;          // // Nombre de blocs par dimension (ici suppose N % B == 0)
  logger.info("Problem: N=" + std::to_string(N) +
              " B=" + std::to_string(B) +
              " Nb=" + std::to_string(Nb));

  // -------------------- Ouverture d'une session --------------------
  // On précise la/les partitions autorisées (ici, CPU)
  std::string session_id = sessionsClient.create_session(taskOptions, {part_cpu});
  logger.info("Session id = " + session_id);

  // -------------------- Enregistrement des IDs de résultats (blocs) --------------------
  // On crée les "résultats" côté Core (clé logique → ID réel) pour le triangle inférieur (j ≤ i)
  std::vector<std::string> all_block_names;
  all_block_names.reserve(Nb*Nb);
  for (int i=0;i<Nb;++i)
    for (int j=0;j<=i;++j)
      all_block_names.push_back(blk_id(i,j));
  // Retour : map nom → resultId (utilisés ensuite pour upload/download)
  std::map<std::string,std::string> id_map =
      resultsClient.create_results_metadata(session_id, all_block_names);

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

    // ---- 1) POTRF(k,k) : factorise la diagonale ----
    {
      // Payload pour le worker : op + indices + IDs in/out
      std::string pl = make_payload_potrf(k, B);

      // Comme ArmoniK attend un "payload" sous forme de Result,
      // on crée un ID pour ce payload et on y "upload" la chaîne pl
      std::string payload_name = "payload/potrf/" + std::to_string(k);
      auto meta = resultsClient.create_results_metadata(session_id, {payload_name});
      resultsClient.upload_result_data(session_id, meta[payload_name], pl);

      // Soumission de la tâche : payloadId + résultat attendu (blk/k/k)
      tasksClient.submit_tasks(session_id, {
        ak_common::TaskCreation{ meta[payload_name], { id_map[blk_id(k,k)] } }
      });

      // Synchronisation : on attend que le bloc diagonal factorisé soit disponible
      eventsClient.wait_for_result_availability(session_id, { id_map[blk_id(k,k)] });
    }

    // ---- 2) TRSM(i,k) pour i>k : "descente" de la colonne k ----
    std::vector<std::string> trsm_out_ids;   // // pour attendre en lot
    for (int i=k+1; i<Nb; ++i) {
      std::string pl = make_payload_trsm(i, k, B);
      std::string payload_name = "payload/trsm/" + std::to_string(i) + "/" + std::to_string(k);
      auto meta = resultsClient.create_results_metadata(session_id, {payload_name});
      resultsClient.upload_result_data(session_id, meta[payload_name], pl);

      tasksClient.submit_tasks(session_id, {
        ak_common::TaskCreation{ meta[payload_name], { id_map[blk_id(i,k)] } }
      });

      trsm_out_ids.push_back(id_map[blk_id(i,k)]);
    }
    // Attente de tous les TRSM de la colonne k
    if (!trsm_out_ids.empty())
      eventsClient.wait_for_result_availability(session_id, trsm_out_ids);

    // ---- 3) MAJ(i,j,k) pour k<j≤i : mises à jour du trailing submatrix ----
    //      - SYRK pour les blocs diagonaux (i==j)
    //      - GEMM pour les blocs hors diagonale (i>j)
    std::vector<std::string> upd_out_ids;
    for (int i=k+1; i<Nb; ++i) {
      for (int j=k+1; j<=i; ++j) {
        const bool is_diag = (i==j);

        // Payload selon le type d'opération
        std::string pl = is_diag ? make_payload_syrk(i, k, B)
                                 : make_payload_gemm(i, j, k, B);

        // ID pour stocker la payload de cette tâche
        std::string payload_name = std::string("payload/") + (is_diag ? "syrk/" : "gemm/")
                                 + std::to_string(i) + "/" + std::to_string(j)
                                 + "/" + std::to_string(k);
        auto meta = resultsClient.create_results_metadata(session_id, {payload_name});
        resultsClient.upload_result_data(session_id, meta[payload_name], pl);

        // Soumission de la tâche de mise à jour qui doit produire blk(i,j)
        tasksClient.submit_tasks(session_id, {
          ak_common::TaskCreation{ meta[payload_name], { id_map[blk_id(i,j)] } }
        });

        upd_out_ids.push_back(id_map[blk_id(i,j)]);
      }
    }
    // Attente de toutes les mises à jour de la vague k
    if (!upd_out_ids.empty())
      eventsClient.wait_for_result_availability(session_id, upd_out_ids);

    logger.info("Wave k=" + std::to_string(k) + " done.");
  }

  // -------------------- Fin --------------------
  logger.info("All waves completed. L factor is in blk/i/j (i≥j).");
  return 0;
}
