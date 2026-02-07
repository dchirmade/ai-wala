"""
ZTE Router Web Dashboard - Flask Application
Real-time signal monitoring with WebSocket and Ollama AI integration
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import time

from zte_router import ZTERouter
from ollama_advisor import OllamaAdvisor

app = Flask(__name__)
app.secret_key = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
routers = {}  # session_id -> ZTERouter
polling_threads = {}  # session_id -> thread
ollama_advisor = OllamaAdvisor(model="llama3")


def get_router(sid: str) -> ZTERouter:
    """Get router instance for session"""
    return routers.get(sid)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/connect', methods=['POST'])
def connect():
    """Connect and login to router"""
    data = request.json
    ip_address = data.get('ip_address', '192.168.0.1')
    password = data.get('password', '')
    sid = request.headers.get('X-Session-ID', 'default')

    if not password:
        return jsonify({"success": False, "error": "Password required"})

    try:
        router = ZTERouter(ip_address)
        print(f"[DEBUG] Connecting to {ip_address}...")
        result = router.login(password)
        print(f"[DEBUG] Login result: {result}")

        if result.get("success"):
            routers[sid] = router
            # Get initial signal info
            signal_info = router.get_signal_info()
            print(f"[DEBUG] Signal info: {signal_info}")
            return jsonify({
                "success": True,
                "signal_info": signal_info,
                "version": router.get_version_info()
            })
        else:
            return jsonify(result)

    except Exception as e:
        print(f"[DEBUG] Exception: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/disconnect', methods=['POST'])
def disconnect():
    """Disconnect from router"""
    sid = request.headers.get('X-Session-ID', 'default')
    if sid in routers:
        del routers[sid]
    if sid in polling_threads:
        polling_threads[sid] = False
    return jsonify({"success": True})


@app.route('/api/signal', methods=['GET'])
def get_signal():
    """Get current signal info"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)

    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    try:
        signal_info = router.get_signal_info()
        return jsonify({"success": True, "signal_info": signal_info})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/network_mode', methods=['POST'])
def set_network_mode():
    """Set network mode"""
    sid = request.headers.get('X-Session-ID', 'default')
    print(f"[DEBUG] Request for network_mode from SID: {sid}")
    router = get_router(sid)
    if not router:
        print(f"[DEBUG] Router not found for SID: {sid}. Available: {list(routers.keys())}")
        return jsonify({"success": False, "error": "Not connected"})

    mode = request.json.get('mode')
    if not mode:
        return jsonify({"success": False, "error": "Mode required"})

    print(f"[DEBUG] Setting network mode to: {mode}")
    result = router.set_network_mode(mode)
    print(f"[DEBUG] Mode set result: {result}")
    return jsonify(result)


@app.route('/api/lte_bands', methods=['POST'])
def set_lte_bands():
    """Set LTE bands"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    bands = request.json.get('bands', 'AUTO')
    print(f"[DEBUG] Setting LTE bands: {bands}")
    result = router.set_lte_bands(bands)
    print(f"[DEBUG] LTE band set result: {result}")
    return jsonify(result)


@app.route('/api/nr_bands', methods=['POST'])
def set_nr_bands():
    """Set 5G NR bands"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    bands = request.json.get('bands', 'AUTO')
    print(f"[DEBUG] Setting NR bands: {bands}")
    result = router.set_nr_bands(bands)
    print(f"[DEBUG] NR band set result: {result}")
    return jsonify(result)


@app.route('/api/lte_cell_lock', methods=['POST'])
def lock_lte_cell():
    """Lock LTE cell"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    data = request.json
    pci = int(data.get('pci', 0))
    earfcn = int(data.get('earfcn', 0))
    result = router.lock_lte_cell(pci, earfcn)
    return jsonify(result)


@app.route('/api/nr_cell_lock', methods=['POST'])
def lock_nr_cell():
    """Lock 5G NR cell"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    data = request.json
    pci = int(data.get('pci', 0))
    arfcn = int(data.get('arfcn', 0))
    band = int(data.get('band', 0))
    scs = int(data.get('scs', 30))
    result = router.lock_nr_cell(pci, arfcn, band, scs)
    return jsonify(result)


@app.route('/api/bridge_mode', methods=['POST'])
def set_bridge_mode():
    """Enable/disable bridge mode"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    enable = request.json.get('enable', False)
    result = router.set_bridge_mode(enable)
    return jsonify(result)


@app.route('/api/arp_proxy', methods=['POST'])
def set_arp_proxy():
    """Enable/disable ARP proxy"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    enable = request.json.get('enable', False)
    result = router.set_arp_proxy(enable)
    return jsonify(result)


@app.route('/api/reboot', methods=['POST'])
def reboot():
    """Reboot router"""
    sid = request.headers.get('X-Session-ID', 'default')
    router = get_router(sid)
    if not router:
        return jsonify({"success": False, "error": "Not connected"})

    result = router.reboot()
    return jsonify(result)


@app.route('/api/ollama/models', methods=['GET'])
def list_ollama_models():
    """List available Ollama models (for login screen)"""
    try:
        import requests as req
        response = req.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Remove duplicates and sort
            model_names = sorted(list(set(model_names)))
            return jsonify({"success": True, "models": model_names})
        return jsonify({"success": False, "models": [], "error": "Ollama not responding"})
    except Exception as e:
        return jsonify({"success": False, "models": [], "error": str(e)})


@app.route('/api/ollama/check', methods=['GET'])
def check_ollama():
    """Check Ollama availability"""
    result = ollama_advisor.check_connection()
    return jsonify(result)


@app.route('/api/ollama/model', methods=['POST'])
def set_ollama_model():
    """Set Ollama model"""
    global ollama_advisor
    model = request.json.get('model', 'llama3')
    print(f"[DEBUG] Switching Ollama model to: {model}")
    ollama_advisor = OllamaAdvisor(model=model)
    # Perform immediate availability check
    check = ollama_advisor.check_connection()
    print(f"[DEBUG] Model check result: {check}")
    return jsonify({"success": True, "model": model, "status": check})


@app.route('/api/ollama/recommend', methods=['POST'])
def get_recommendation():
    """Get AI band optimization recommendation"""
    sid = request.headers.get('X-Session-ID', 'default')
    print(f"[DEBUG] Request for recommendation from SID: {sid}")
    router = get_router(sid)
    if not router:
        print(f"[DEBUG] Router not found for SID: {sid}. Available: {list(routers.keys())}")
        return jsonify({"success": False, "error": "Not connected to router"})

    try:
        signal_info = router.get_signal_info()
        recommendation = ollama_advisor.get_recommendation(signal_info)

        if recommendation.get("success"):
            # Try to parse actionable bands from recommendation
            parsed = ollama_advisor.parse_bands_from_recommendation(
                recommendation.get("recommendation", "")
            )
            recommendation["parsed"] = parsed

        return jsonify(recommendation)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    emit('status', {'connected': True})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    sid = request.sid
    if sid in polling_threads:
        polling_threads[sid] = False


@socketio.on('start_polling')
def start_polling(data):
    """Start signal polling for this client"""
    sid = request.sid
    ip_address = data.get('ip_address', '192.168.0.1')
    password = data.get('password', '')

    if not password:
        emit('error', {'message': 'Password required'})
        return

    # Create router instance
    router = ZTERouter(ip_address)
    result = router.login(password)

    if not result.get("success"):
        emit('error', {'message': result.get('error', 'Login failed')})
        return

    routers[sid] = router
    polling_threads[sid] = True

    emit('connected', {'success': True, 'version': router.get_version_info()})

    # Start polling thread
    def poll_signal():
        while polling_threads.get(sid, False):
            try:
                signal_info = router.get_signal_info()
                if "net_select" in router._signal_data:
                    print(f"[DEBUG] CURRENT MODE FIELDS - net_select: {router._signal_data.get('net_select')}, BearerPreference: {router._signal_data.get('BearerPreference')}, network_type: {router._signal_data.get('network_type')}")
                socketio.emit('signal_update', signal_info, room=sid)
            except Exception as e:
                socketio.emit('error', {'message': str(e)}, room=sid)
            time.sleep(1)

    thread = threading.Thread(target=poll_signal, daemon=True)
    thread.start()


@socketio.on('stop_polling')
def stop_polling():
    """Stop signal polling"""
    sid = request.sid
    polling_threads[sid] = False
    if sid in routers:
        del routers[sid]


if __name__ == '__main__':
    print("=" * 60)
    print("ZTE Router Dashboard - Python + Ollama Edition")
    print("=" * 60)
    print("\nStarting server on http://localhost:5000")
    print("\nMake sure Ollama is running: ollama serve")
    print("=" * 60)

    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True, use_reloader=False)
