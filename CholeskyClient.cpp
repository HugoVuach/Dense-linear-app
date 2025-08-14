// ============================================================================
//  Client ArmoniK pour lancer un worker "Cholesky" (version monolithique)
// dans le sens où :
// Pas de découpage du calcul en plusieurs tâches ArmoniK pour différents blocs.
// Pas de DAG distribué (POTRF, TRSM, SYRK/GEMM séparés).
// Le découpage en tuiles est interne à Chameleon/StarPU sur un seul nœud.
// ArmoniK ici ne sert qu’à orchestrer l’exécution du worker, pas à répartir 
// les blocs entre plusieurs nœuds.
// [Client] --(1 tâche + petite payload)--> [ArmoniK] --(planifie)--> [Worker unique]
//                                                         |
//                                                         v
//                                                 [Chameleon(+StarPU) interne]
//                                                -> fait TOUTE la Cholesky


//  - Envoie une payload texte : N NB mb nb bsiz lm ln ioff joff m n p q seed
//  - Attend un résultat "summary" que le worker publie après calcul
// ============================================================================

#include "objects.pb.h"                 // // Messages Protobuf (définitions gRPC)
#include "utils/Configuration.h"        // // Gestion de la configuration (JSON, ENV)
#include "logger/logger.h"              // // Logger ArmoniK (interface)
#include "logger/writer.h"              // // Writer (console, fichier)
#include "logger/formatter.h"           // // Format du logging (plain/json)
#include "channel/ChannelFactory.h"     // // Usine à canaux gRPC selon la config
#include "sessions/SessionsClient.h"    // // Client gRPC pour la gestion des sessions
#include "tasks/TasksClient.h"          // // Client gRPC pour la soumission de tâches
#include "results/ResultsClient.h"      // // Client gRPC pour la gestion des résultats
#include "events/EventsClient.h"        // // Client gRPC pour s'abonner/attendre des événements

#include <map>                          // // std::map pour manipuler les IDs de résultats
#include <string>                       // // std::string
#include <sstream>                      // // std::ostringstream pour construire la payload
#include <vector>                       // // std::vector (utilisé par certaines API)
#include <iostream>                     // // (optionnel) std::cout si besoin

// Alias de noms pour raccourcir le code
namespace ak_common = armonik::api::common;   // // Espace commun (types utilitaires, TaskCreation, etc.)
namespace ak_client = armonik::api::client;   // // Espace des clients (SessionsClient, TasksClient, ...)
namespace ak_grpc   = armonik::api::grpc::v1; // // Espace des stubs gRPC (protos v1)

int main() {
  // --------------------------------------------------------------------------
  // 1) Initialisation du logger et de la configuration
  // --------------------------------------------------------------------------
  ak_common::logger::Logger logger{
    ak_common::logger::writer_console(),          // // Sortie des logs sur la console
    ak_common::logger::formatter_plain(true)      // // Format "plain" (lisible) + horodatage
  };

  ak_common::utils::Configuration config;         // // Objet de config (clé-valeur)
  // Charge d'abord /appsettings.json, puis surcharge avec les variables d'environnement
  config.add_json_configuration("/appsettings.json").add_env_configuration();
  logger.info("Initialized client config.");

  // --------------------------------------------------------------------------
  // 2) Création du canal gRPC vers ArmoniK (Gateway)
  // --------------------------------------------------------------------------
  ak_client::ChannelFactory channelFactory(config, logger); // // Fabrique qui lit la config (adresse, TLS, etc.)
  std::shared_ptr<::grpc::Channel> channel = channelFactory.create_channel(); // // Canal gRPC prêt

  // --------------------------------------------------------------------------
  // 3) Préparation des options de tâche (TaskOptions) et de la session
  // --------------------------------------------------------------------------
  ak_grpc::TaskOptions taskOptions;

  // Nom de la partition à utiliser (doit correspondre à celle où ton worker est déployé)
  std::string used_partition = "cholesky-cpu";
  logger.info("Using the '" + used_partition + "' partition.");

  // Paramètres généraux de la tâche
  taskOptions.mutable_max_duration()->set_seconds(3600); // // Durée max (1h)
  taskOptions.set_max_retries(3);                        // // Jusqu'à 3 tentatives en cas d'échec
  taskOptions.set_priority(1);                           // // Priorité relative (1 = défaut)
  taskOptions.set_partition_id(used_partition);          // // Partition ciblée (routing côté ArmoniK)

  // Métadonnées "app" (libres, utiles pour traçabilité/monitoring)
  taskOptions.set_application_name("cholesky-cpp");      // // Nom logique de l'app
  taskOptions.set_application_version("1.0");            // // Version applicative
  taskOptions.set_application_namespace("benchmarks");   // // Espace de nom logique

  // --------------------------------------------------------------------------
  // 4) Construction des clients ArmoniK (Sessions / Tasks / Results / Events)
  // --------------------------------------------------------------------------
  ak_client::TasksClient    tasksClient(   ak_grpc::tasks::Tasks::NewStub(channel));
  ak_client::ResultsClient  resultsClient( ak_grpc::results::Results::NewStub(channel));
  ak_client::SessionsClient sessionsClient(ak_grpc::sessions::Sessions::NewStub(channel));
  ak_client::EventsClient   eventsClient(  ak_grpc::events::Events::NewStub(channel));

  // --------------------------------------------------------------------------
  // 5) Création d'une session d'exécution
  // --------------------------------------------------------------------------
  // On crée une session qui encapsule nos soumissions. On précise la/les partitions
  // autorisées (ici, juste "cholesky-cpu").
  std::string session_id = sessionsClient.create_session(taskOptions, {used_partition});
  logger.info("Created session with id = " + session_id);

  // --------------------------------------------------------------------------
  // 6) Construire la payload attendue par le worker
  // --------------------------------------------------------------------------
  // Ordre exact des 14 entiers : N NB mb nb bsiz lm ln ioff joff m n p q seed
  // ⚠️ 'ncpu' et 'ngpu' ne sont PAS dans la payload : ils sont lus côté worker via l'ENV
  int N    = 4096;   // // Taille globale (N x N)
  int NB   = 256;    // // Taille de tuile "algorithmique" (si utilisée par Chameleon pour dpotrf)
  int mb   = 256;    // // Hauteur de tuile pour la desc (descA)
  int nb   = 256;    // // Largeur  de tuile pour la desc (descA)
  int bsiz = mb * nb; // // Produit (contrôle de cohérence)
  int lm   = 4096;   // // Leading dimension globale (nombre de lignes logiques)
  int ln   = 4096;   // // Leading dimension globale (nombre de colonnes logiques)
  int ioff = 0;      // // Décalage de bloc (ligne) dans la desc (souvent 0)
  int joff = 0;      // // Décalage de bloc (colonne) dans la desc (souvent 0)
  int m    = 4096;   // // Sous-matrice M (lignes) gérée par la desc (souvent = lm)
  int n    = 4096;   // // Sous-matrice N (colonnes) gérée par la desc (souvent = ln)
  int p    = 1;      // // Découpage processus (rangées) si tu utilises un grid p x q (ici 1x1)
  int q    = 1;      // // Découpage processus (colonnes)
  int seed = 2025;   // // Graine pour générer la SPD (dplgsy)

  // Construction de la payload texte (séparateurs ' ')
  std::ostringstream payload_oss;
  payload_oss << N << ' ' << NB << ' ' << mb << ' ' << nb << ' ' << bsiz << ' '
              << lm << ' ' << ln << ' ' << ioff << ' ' << joff << ' '
              << m  << ' ' << n  << ' ' << p   << ' ' << q    << ' ' << seed;
  const std::string payload_text = payload_oss.str();

  // --------------------------------------------------------------------------
  // 7) Créer des métadonnées de résultats (IDs gérés par ArmoniK.Core)
  // --------------------------------------------------------------------------
  // On demande 2 IDs : 
  //  - "payload" : pour stocker la payload d'entrée (petite)
  //  - "summary" : résultat attendu (texte renvoyé par le worker : Time, Gflop/s, Error, ...)
  std::map<std::string,std::string> results =
      resultsClient.create_results_metadata(session_id, {"summary", "payload"});

  // --------------------------------------------------------------------------
  // 8) Uploader la payload dans le store de résultats
  // --------------------------------------------------------------------------
  // On écrit 'payload_text' dans l'ID 'results["payload"]'. La tâche référencera cet ID.
  resultsClient.upload_result_data(session_id, results["payload"], payload_text);
  logger.info("Uploaded payload for Cholesky run.");

  // --------------------------------------------------------------------------
  // 9) Soumettre la tâche
  // --------------------------------------------------------------------------
  // Le modèle : TaskCreation{ payload_result_id, {expected_result_ids...} }
  // → payload_result_id  : ID du blob "payload" (que le worker lira via getPayload())
  // → expected_result_ids: liste d'IDs que le worker devra remplir (ici "summary")
  auto submissions = tasksClient.submit_tasks(
      session_id,
      { ak_common::TaskCreation{ results["payload"], { results["summary"] } } }
  );
  const auto &task_info = submissions[0];
  logger.info("Task submitted with id = " + task_info.task_id());

  // --------------------------------------------------------------------------
  // 10) Attendre que le résultat "summary" soit disponible
  // --------------------------------------------------------------------------
  // L'EventsClient permet de se bloquer jusqu'à la disponibilité d'un résultat donné.
  logger.info("Waiting for result id = " + results["summary"]);
  eventsClient.wait_for_result_availability(session_id, { results["summary"] });
  logger.info("Result is now available.");

  // --------------------------------------------------------------------------
  // 11) Télécharger et afficher le résultat
  // --------------------------------------------------------------------------
  // Le worker renvoie un texte multi-lignes (N, NB, Time, Perf, RelError, Status)
  std::string taskResult = resultsClient.download_result_data(session_id, { results["summary"] });
  logger.info("Worker summary:\n" + taskResult);

  // --------------------------------------------------------------------------
  // 12) Fin du programme
  // --------------------------------------------------------------------------
  return 0;
}
