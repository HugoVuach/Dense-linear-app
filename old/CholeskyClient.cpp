// ============================================================================
//  Client ArmoniK pour lancer un worker "Cholesky" monolithique
// Pas de découpage du calcul en plusieurs tâches ArmoniK pour différents blocs.
// Pas de DAG distribué.
// Le découpage en tuiles est interne à Chameleon/StarPU sur un seul nœud.
// ArmoniK ici ne sert qu’à orchestrer l’exécution du worker, pas à répartir 
// les blocs entre plusieurs nœuds.
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



// Alias de noms pour raccourcir le code
namespace ak_common = armonik::api::common;   
namespace ak_client = armonik::api::client;   
namespace ak_grpc   = armonik::api::grpc::v1; 

int main() {
  // --------------------------------------------------------------------------
  // 1) Initialisation du logger et de la configuration
  // --------------------------------------------------------------------------
  ak_common::logger::Logger logger{ak_common::logger::writer_console(), ak_common::logger::formatter_plain(true)};
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

  // Nom de la partition à utiliser ici default
  std::string used_partition = "default";
  logger.info("Using the '" + used_partition + "' partition.");

  // Paramètres généraux de la tâche
  taskOptions.mutable_max_duration()->set_seconds(3600); // Durée max 
  taskOptions.set_max_retries(3);                        // 3 tentatives si échec
  taskOptions.set_priority(1);                           // Priorité 
  taskOptions.set_partition_id(used_partition);          // Partition ciblée

  // Métadonnées
  taskOptions.set_application_name("cholesky-cpp");      // Nom logique de l'app
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
  // 5) Création d'une session d'exécution
  // --------------------------------------------------------------------------
  // On crée une session qui encapsule nos soumissions. On précise la/les partitions
  // autorisées (ici, juste "default").
  std::string session_id = sessionsClient.create_session(taskOptions, {used_partition});
  logger.info("Created session with id = " + session_id);

  // --------------------------------------------------------------------------
  // 6) Construire la payload attendue par le worker
  // --------------------------------------------------------------------------
  int N    = 4096;   // Taille Matrice
  int NB   = 256;    // Taille de tuile 
  int mb   = 256;    // Hauteur de tuile pour la desc (descA)
  int nb   = 256;    // Largeur  de tuile pour la desc (descA)
  int bsiz = mb * nb;// Produit contrôle de cohérence
  int lm   = 4096;   // Leading dimension globale 
  int ln   = 4096;   // Leading dimension globale
  int ioff = 0;      // Décalage de bloc (ligne) dans la desc 
  int joff = 0;      // Décalage de bloc (colonne) dans la desc 
  int m    = 4096;   // Sous-matrice M (lignes) gérée par la desc (souvent = lm)
  int n    = 4096;   // Sous-matrice N (colonnes) gérée par la desc (souvent = ln)
  int p    = 1;      // Découpage processus (rangées) 
  int q    = 1;      // Découpage processus (colonnes)
  int seed = 2025;   // Graine pour générer la SPD 

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
  //  - "summary" : résultat attendu (texte renvoyé par le worker : Time, Gflop/s, Error, ...)
  //  - "payload" : pour stocker la payload d'entrée (petite)
  std::map<std::string,std::string> results = resultsClient.create_results_metadata(session_id, {"summary", "payload"});

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
  auto submissions = tasksClient.submit_tasks(session_id,{ ak_common::TaskCreation{ results["payload"], { results["summary"] } } });
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
  // 11) Télécharger et afficher le résultat du worker et fin du programme
  //              (N, NB, Time, Perf, RelError, Status)
  // --------------------------------------------------------------------------
  std::string taskResult = resultsClient.download_result_data(session_id, { results["summary"] });
  logger.info("Worker summary:\n" + taskResult);
  return 0;
}
