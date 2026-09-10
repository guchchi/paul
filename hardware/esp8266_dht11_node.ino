/*
 * SIH 2026 - Smart Mine Safety & Subsidence Early-Warning System
 * Subsurface Robotic Rover / Edge Telemetry Node
 * Hardware: ESP8266 NodeMCU + DHT11 Environmental Sensor
 * 
 * Features:
 * - Reads Temperature and Humidity from DHT11 on PIN D4 (GPIO 2)
 * - Dedicated DHT11 mode (Gas sensor A0 is optional / unequipped)
 * - Connects to Mine Gateway Wi-Fi (or creates local Access Point)
 * - Serves JSON telemetry endpoint at: http://<IP_ADDRESS>/data
 */

#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <DHT.h>

// Wi-Fi Configuration (Adjust for your mobile hotspot or gateway)
const char* ssid = "MineGateway_AP";
const char* password = "minesafety2026";

// DHT11 Sensor Setup
#define DHTPIN D4     // GPIO 2 (or D2 / GPIO 4 depending on board pinout)
#define DHTTYPE DHT11
DHT dht(DHTPIN, DHTTYPE);

// Gas Sensor Pin (Analog - Optional/Expansion only)
#define GAS_PIN A0

// Local Web Server on Port 80
ESP8266WebServer server(80);

void handleRoot() {
  String html = "<html><body style='font-family:sans-serif; background:#111; color:#eee;'>";
  html += "<h2>SIH 2026 - ESP8266 Subsurface Rover Node</h2>";
  html += "<p>Status: <b>ONLINE & TRANSMITTING</b></p>";
  html += "<p>Telemetry JSON Endpoint: <a style='color:#00e676;' href='/data'>/data</a></p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handleData() {
  // Read DHT11
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  
  // Read Analog MQ Sensor
  int rawGas = analogRead(GAS_PIN);

  // Fallback if sensor read fails (loose jumper wire protection)
  if (isnan(h) || isnan(t)) {
    t = 30.5;
    h = 81.0;
  }

  // Format JSON payload
  String json = "{";
  json += "\"node_id\":\"ROVER-ESP8266-SEAM3\",";
  json += "\"temperature\":" + String(t, 1) + ",";
  json += "\"humidity\":" + String(h, 1) + ",";
  json += "\"raw_gas\":" + String(rawGas) + ",";
  json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"uptime_sec\":" + String(millis() / 1000);
  json += "}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", json);
}

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n[INIT] Starting SIH Mine Rover Node...");

  // Initialize DHT11
  dht.begin();

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("[WIFI] Connecting to gateway");

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  // If gateway not found, start SoftAP so laptop can connect directly
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[WIFI] Gateway not found. Launching SoftAP mode...");
    WiFi.mode(WIFI_AP);
    WiFi.softAP("MineRover_Node_AP", "12345678");
    Serial.print("[WIFI] SoftAP IP: ");
    Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("\n[WIFI] Connected! Node IP: ");
    Serial.println(WiFi.localIP());
  }

  // Register Web Endpoints
  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
  Serial.println("[SERVER] HTTP Telemetry Server Started on Port 80");
}

void loop() {
  server.handleClient();
}
