/*
  IntelliSignal ESP32 - Single Signal Display
  ---------------------------------------------
  Only 3 LEDs total (Red, Yellow, Green)
  Shows the priority line's signal state when emergency detected on ANY line.

  When emergency detected on any line (1-6):
    → Green LED ON for 60s
    → Yellow LED ON for 5s  
    → Red LED ON (normal)

  Wiring:
    D2  → 220Ω resistor → Red LED(+)    → LED(-) → GND
    D15 → 220Ω resistor → Yellow LED(+) → LED(-) → GND
    D18 → 220Ω resistor → Green LED(+)  → LED(-) → GND
    GND → long horizontal rail on breadboard edge

  Libraries (Arduino IDE → Tools → Manage Libraries):
    - WebSockets  by Markus Sattler
    - ArduinoJson by Benoit Blanchon
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// ── Change these 3 values ────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_NAME";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_HOST   = "192.168.1.100";  // your PC IP from ipconfig
// ─────────────────────────────────────────────────────────────────────────────

const int SERVER_PORT = 5000;

// 3 LED pins
const int RED_PIN    = 2;
const int YELLOW_PIN = 15;
const int GREEN_PIN  = 18;

bool   wsConnected = false;
unsigned long lastPollMs = 0;
const unsigned long POLL_INTERVAL_MS = 5000;

WebSocketsClient ws;

// ── Set signal state ──────────────────────────────────────────────────────────
void setSignal(String state) {
    digitalWrite(RED_PIN,    state == "red"    ? HIGH : LOW);
    digitalWrite(YELLOW_PIN, state == "yellow" ? HIGH : LOW);
    digitalWrite(GREEN_PIN,  state == "green"  ? HIGH : LOW);
    Serial.println("[Signal] → " + state.toUpperCase());
}

// ── WebSocket handler ─────────────────────────────────────────────────────────
void onWsEvent(WStype_t type, uint8_t* payload, size_t length) {
    switch (type) {

        case WStype_CONNECTED:
            wsConnected = true;
            Serial.println("[WS] Connected to Flask");
            ws.sendTXT("40");  // Socket.IO connect
            break;

        case WStype_DISCONNECTED:
            wsConnected = false;
            Serial.println("[WS] Disconnected — polling fallback active");
            break;

        case WStype_TEXT: {
            String msg = String((char*)payload);

            // Heartbeat ping → pong
            if (msg == "2") { ws.sendTXT("3"); break; }
            if (!msg.startsWith("42")) break;

            String json = msg.substring(2);
            StaticJsonDocument<512> doc;
            if (deserializeJson(doc, json) != DeserializationError::Ok) break;

            String event = doc[0].as<String>();

            if (event == "signal_changed") {
                String state    = doc[1]["state"].as<String>();
                bool emergency  = doc[1]["emergency"].as<bool>();
                String lineId   = doc[1]["line_id"].as<String>();
                String priority = doc[1]["priority_line"].as<String>();

                // Only react to the priority line (the one with emergency)
                if (emergency && lineId == priority) {
                    setSignal(state);
                    if (state == "green")
                        Serial.println("[Emergency] " + lineId + " → GREEN for 60s");
                    else if (state == "yellow")
                        Serial.println("[Emergency] " + lineId + " → YELLOW transitioning");
                    else if (state == "red")
                        Serial.println("[Emergency] " + lineId + " → RED (sequence done)");
                }
            }
            break;
        }

        default: break;
    }
}

// ── HTTP fallback polling ─────────────────────────────────────────────────────
void pollSignalStates() {
    if (WiFi.status() != WL_CONNECTED) return;

    HTTPClient http;
    String url = String("http://") + SERVER_HOST + ":" + SERVER_PORT + "/api/dashboard/signals";
    http.begin(url);
    http.setTimeout(3000);

    int code = http.GET();
    if (code == 200) {
        String body = http.getString();
        StaticJsonDocument<512> doc;
        if (deserializeJson(doc, body) == DeserializationError::Ok) {
            // Find any line that is green (emergency active)
            bool foundGreen = false;
            for (int i = 1; i <= 6; i++) {
                String key   = "line-" + String(i);
                String state = doc[key].as<String>();
                if (state == "green") {
                    setSignal("green");
                    foundGreen = true;
                    break;
                } else if (state == "yellow") {
                    setSignal("yellow");
                    foundGreen = true;
                    break;
                }
            }
            if (!foundGreen) setSignal("red");
            Serial.println("[Poll] States synced");
        }
    }
    http.end();
}

// ── Setup ─────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    pinMode(RED_PIN,    OUTPUT);
    pinMode(YELLOW_PIN, OUTPUT);
    pinMode(GREEN_PIN,  OUTPUT);
    setSignal("red");  // default red on boot

    // Connect WiFi
    Serial.printf("\nConnecting to %s", WIFI_SSID);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500); Serial.print(".");
    }
    Serial.println("\nWiFi OK — IP: " + WiFi.localIP().toString());

    // Sync current state from server
    pollSignalStates();

    // Connect WebSocket
    ws.begin(SERVER_HOST, SERVER_PORT,
             "/socket.io/?EIO=4&transport=websocket");
    ws.onEvent(onWsEvent);
    ws.setReconnectInterval(3000);
    Serial.println("[WS] Connecting...");
}

// ── Loop ──────────────────────────────────────────────────────────────────────
void loop() {
    ws.loop();

    unsigned long now = millis();
    if (!wsConnected && (now - lastPollMs >= POLL_INTERVAL_MS)) {
        lastPollMs = now;
        pollSignalStates();
    }
}
