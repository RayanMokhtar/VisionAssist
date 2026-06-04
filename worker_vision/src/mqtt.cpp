#include "mqtt.h"  
#include <mqtt/connect_options.h>  
#include <iostream>  //logs



//documentation https://github.com/eclipse-paho/paho.mqtt.cpp
// edit : si lancement du code sur wsl , et que broker sur windows dans ce cas il faut fetch d'abord l'ip du broker 


MqttClient::MqttClient(const std::string& host, int port, const std::string& clientId):
    brokerAddress("tcp://" + host + ":" + std::to_string(port)), 
    clientId(clientId), 
    client(brokerAddress, clientId) //initialisation du client
{
    try {
        mqtt::connect_options connOpts;
        connOpts.set_keep_alive_interval(60);  
        connOpts.set_clean_session(true);
        client.connect(connOpts);
        std::cout << "MQTT connecte à " << brokerAddress << std::endl;
    } catch (const mqtt::exception& exc) {
        std::cerr << "MQTT connexion pas établie: " << exc.what() << std::endl;
    }
}


MqttClient::~MqttClient() {
    deconnexion();
}

bool MqttClient::estConnecte() {
    return client.is_connected();
}

void MqttClient::deconnexion() {
    try {
        if (client.is_connected()) {
            client.disconnect();
            std::cout << "MQTT deconnecte." << std::endl;
        }
    } catch (const mqtt::exception& exc) {
        std::cerr << "MQTT eerreur deconnexion dans méthode deconnexion: " << exc.what() << std::endl;
    }
}

bool MqttClient::publier(const std::string& topic, const std::string& message, const int qos, const bool retain) { //je sais pas si const message ici est approprié à cause du c_str ? application des modifs sur elle même ou que sur résutlat de la fonciton
    try {
        const char* messageEnPointeur = message.c_str(); //paho mqtt dans la doc eplique qu'on attend un message pointeur // First use a message pointer : mqtt::message_ptr pubmsg = mqtt::make_message(PAYLOAD1);
        client.publish(topic, messageEnPointeur, message.length(), qos, retain); 
        std::cout << "publié au topic :  " << topic << " message =>  " << message << std::endl;
        return true;
    } catch (const mqtt::exception& exc) {
        std::cerr << "MQTT publicaiton a échoué : " << exc.what() << std::endl;
        return false;
    }
}

bool MqttClient::sabonner(const std::string& topic , const int qos) {  
    try {
        client.subscribe(topic, qos);  //par défaut ce sera 1
        std::cout << "abonnement réussi au topic :  " << topic << std::endl;
        return true;
    } catch (const mqtt::exception& exc) {
        std::cerr << "MQaTT abonnement échoué => cause => : " << exc.what() << std::endl;
        return false;
    }
}