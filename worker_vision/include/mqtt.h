#ifndef MQTT_H 
#define MQTT_H 


#include <mqtt/client.h>
#include <string.h>

class MqttClient {

public : 
    MqttClient(const std::string& host = "192.168.0.21", int port = 1883 , const std::string& clientId="worker_vision");//constructeur
    ~MqttClient();//destructeur
    bool estConnecte();
    void deconnexion();
    bool publier(const std::string& topic, const std::string& message, const int qos, const bool retain);
    bool sabonner(const std::string& topic, const int qos); // référence mieux que pointeur

private : 
    std::string brokerAddress;
    std::string clientId;
    mqtt::client client; 

};

#endif  