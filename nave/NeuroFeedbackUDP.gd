extends Node

# ==============================================================================
# CONFIGURACIÓN UDP OPTIMIZADA
# ==============================================================================
var udp := PacketPeerUDP.new()
var is_listening := false

# Configuración de puertos
const PYTHON_UDP_PORT = 9081
const GODOT_UDP_PORT = 9080
const PYTHON_IP = "127.0.0.1"

# Estados del juego - OPTIMIZADO
var current_game_state: String = "starting"
var current_energy: int = 0
var current_mission_state: String = "critical"  # Iniciar como "critical"

# Referencias a minijuegos - MODIFICADO: Ahora se limpian automáticamente
var corsi_game: Node = null
var nback_game: Node = null

# Neurofeedback
var current_ratio: float = 0.5  # Valor inicial

# Control de frecuencia de envío - NUEVO
var last_energy_sent: int = -1
var last_game_state_sent: String = ""
var last_mission_state_sent: String = ""
var send_cooldown: float = 0.0
const MIN_SEND_INTERVAL: float = 0.1  # 100ms mínimo entre envíos

# Cache para reducir procesamiento - NUEVO
var current_scene_cache: String = ""
var scene_check_timer: float = 0.0
const SCENE_CHECK_INTERVAL: float = 0.3  # Revisar escena cada 300ms

# NUEVO: Sistema de registro para UI y otros sistemas
var _registered_ui_systems: Array = []
var _registered_minigames: Array = []

# NUEVO: Control para evitar estados residuales de minijuegos
var _minigame_cleanup_timer: float = 0.0
const MINIGAME_CLEANUP_DELAY: float = 2.0  # Limpiar minijuegos después de 2 segundos

func _ready():
	print("🚀 Inicializando sistema de comunicación UDP...")
	start_udp_server()

func start_udp_server():
	"""Inicia el servidor UDP optimizado"""
	var err = udp.bind(GODOT_UDP_PORT, "127.0.0.1")
	if err != OK:
		print("❌ Error al iniciar servidor UDP: ", err)
		return
	
	is_listening = true
	print("✅ Servidor UDP iniciado en puerto ", GODOT_UDP_PORT)
	
	# Enviar estado inicial después de breve delay
	await get_tree().create_timer(0.5).timeout
	_force_send_game_state()

func _process(delta):
	if is_listening:
		_handle_incoming_messages()
	
	# Actualizar cooldown
	send_cooldown = max(0, send_cooldown - delta)
	scene_check_timer += delta
	_minigame_cleanup_timer += delta
	
	# Actualizar energía desde PlayerState (menos frecuente)
	if scene_check_timer >= SCENE_CHECK_INTERVAL:
		_update_energy_from_player_state()
		_detect_scene_changes()
		scene_check_timer = 0.0
	
	# NUEVO: Limpiar referencias a minijuegos periódicamente
	if _minigame_cleanup_timer >= MINIGAME_CLEANUP_DELAY:
		_cleanup_minigame_references()
		_minigame_cleanup_timer = 0.0

func _cleanup_minigame_references():
	"""Limpia referencias a minijuegos que ya no existen - NUEVO"""
	if corsi_game and not is_instance_valid(corsi_game):
		print("🧹 Limpiando referencia a CorsiGame (ya no existe)")
		corsi_game = null
	
	if nback_game and not is_instance_valid(nback_game):
		print("🧹 Limpiando referencia a NBackGame (ya no existe)")
		nback_game = null
	
	# Limpiar arrays de registro
	for i in range(_registered_minigames.size() - 1, -1, -1):
		if not is_instance_valid(_registered_minigames[i]):
			_registered_minigames.remove_at(i)
	
	for i in range(_registered_ui_systems.size() - 1, -1, -1):
		if not is_instance_valid(_registered_ui_systems[i]):
			_registered_ui_systems.remove_at(i)

func _update_energy_from_player_state():
	"""Actualiza energía desde PlayerState de forma optimizada"""
	if has_node("/root/PlayerState"):
		var player_state = get_node("/root/PlayerState")
		var new_energy = player_state.get_total_energy()
		
		if new_energy != current_energy:
			current_energy = new_energy
			print("🔋 UDP - Energía actualizada desde PlayerState: ", current_energy)
			_queue_game_state_send()

func _detect_scene_changes():
	"""Detección optimizada de cambios de escena - VERSIÓN MEJORADA"""
	var new_scene_id = _get_scene_identifier()
	
	if new_scene_id != current_scene_cache:
		current_scene_cache = new_scene_id
		var new_game_state = _determine_game_state(new_scene_id)
		
		# NUEVO: Solo actualizar si realmente cambió el estado
		if new_game_state != current_game_state:
			current_game_state = new_game_state
			print("🎮 UDP - Estado del juego cambiado: ", current_game_state)
			_queue_game_state_send()

func _get_scene_identifier() -> String:
	"""Identificador único y rápido de la escena actual"""
	var current_scene = get_tree().current_scene
	if current_scene == null:
		return "null"
	
	# Usar combinación de nombre y ruta para mayor precisión
	return current_scene.name + "|" + (current_scene.scene_file_path if current_scene.scene_file_path else "no_path")

func _determine_game_state(scene_id: String) -> String:
	"""Determina el estado del juego basado en identificador de escena - VERSIÓN MEJORADA"""
	# NUEVO: Verificación más robusta para evitar falsos positivos
	if "corsi" in scene_id.to_lower() and not "main" in scene_id.to_lower():
		return "in_corsi_minigame"
	elif ("nback" in scene_id.to_lower() or "n-back" in scene_id.to_lower()) and not "main" in scene_id.to_lower():
		return "in_nback_minigame"
	elif "main" in scene_id.to_lower():
		return "exploring"
	elif "menu" in scene_id.to_lower():
		return "in_menu"
	else:
		# Fallback: buscar nodos específicos (solo si es necesario)
		return _fallback_scene_detection()

func _fallback_scene_detection() -> String:
	"""Detección de respaldo por nodos (más costosa, usar solo cuando sea necesario) - VERSIÓN MEJORADA"""
	var current_scene = get_tree().current_scene
	if current_scene:
		# NUEVO: Verificar primero si es la escena principal
		if current_scene.has_node("Player") or current_scene.find_child("Player", true, false):
			return "exploring"
		# Solo si no hay Player, buscar minijuegos
		if current_scene.has_node("CorsiGame") or current_scene.find_child("CorsiGame", true, false):
			return "in_corsi_minigame"
		if current_scene.has_node("NBackGame") or current_scene.find_child("NBackGame", true, false):
			return "in_nback_minigame"
	return "unknown"

func _queue_game_state_send():
	"""Envía estado del juego con control de frecuencia"""
	if send_cooldown <= 0:
		_send_game_state_optimized()

func _send_game_state_optimized():
	"""Envía estado del juego optimizado (solo si hay cambios reales) - VERSIÓN MEJORADA"""
	var should_send = (current_energy != last_energy_sent or 
					  current_game_state != last_game_state_sent or 
					  current_mission_state != last_mission_state_sent)
	
	if should_send:
		var message = {
			"type": "game_state",
			"state": current_game_state,
			"energy": current_energy,
			"mission_state": current_mission_state,
			"timestamp": Time.get_unix_time_from_system()
		}
		_send_to_python(message)
		
		# DEBUG: Mostrar información enviada
		print("📡 UDP - Enviando estado completo: ", message)
		
		# Actualizar cache
		last_energy_sent = current_energy
		last_game_state_sent = current_game_state
		last_mission_state_sent = current_mission_state
		send_cooldown = MIN_SEND_INTERVAL

func _force_send_game_state():
	"""Fuerza el envío del estado actual (para estados importantes)"""
	last_energy_sent = -1  # Reset para forzar envío
	last_game_state_sent = ""
	last_mission_state_sent = ""
	_send_game_state_optimized()

func _handle_incoming_messages():
	"""Maneja mensajes entrantes de forma optimizada"""
	while udp.get_available_packet_count() > 0:
		var packet = udp.get_packet()
		var data = packet.get_string_from_utf8()
		
		var json_data = JSON.new()
		var error = json_data.parse(data)
		
		if error == OK:
			var message = json_data.get_data()
			_process_python_message(message)

func _process_python_message(message: Dictionary):
	"""Procesa mensajes de Python optimizado"""
	var message_type = message.get("type", "")
	
	if message_type == "neurofeedback":
		var new_ratio = message.get("ratio", 0.5)
		current_ratio = clamp(new_ratio, 0.0, 1.0)
		_update_neurofeedback(current_ratio)

func _update_neurofeedback(ratio: float):
	"""Actualiza neurofeedback en todos los sistemas registrados - VERSIÓN MEJORADA"""
	# Notificar a todos los sistemas UI registrados
	for ui_system in _registered_ui_systems:
		if ui_system and is_instance_valid(ui_system) and ui_system.has_method("set_brain_ratio"):
			ui_system.set_brain_ratio(ratio)
	
	# Notificar a todos los minijuegos registrados
	for minigame in _registered_minigames:
		if minigame and is_instance_valid(minigame) and minigame.has_method("set_brain_ratio"):
			minigame.set_brain_ratio(ratio)
	
	# También notificar a los minijuegos específicos (para compatibilidad)
	if corsi_game and is_instance_valid(corsi_game) and corsi_game.has_method("set_brain_ratio"):
		corsi_game.set_brain_ratio(ratio)
	
	if nback_game and is_instance_valid(nback_game) and nback_game.has_method("set_brain_ratio"):
		nback_game.set_brain_ratio(ratio)

func send_minigame_state(minigame_type: String, state: String):
	"""Envía estado de minijuego con prioridad - VERSIÓN MEJORADA"""
	# NUEVO: Solo enviar si realmente estamos en ese minijuego
	if current_game_state == "in_" + minigame_type + "_minigame":
		var message = {
			"type": "minigame_state",
			"minigame_type": minigame_type,
			"state": state,
			"timestamp": Time.get_unix_time_from_system()
		}
		_send_to_python(message)

func send_module_event(module_type: String, event: String, success: bool = true):
	"""Envía eventos de módulo optimizado"""
	var message = {
		"type": "module_event",
		"module_type": module_type,
		"event": event,
		"success": success,
		"timestamp": Time.get_unix_time_from_system()
	}
	_send_to_python(message)

func _send_to_python(message: Dictionary):
	"""Envía mensaje a Python optimizado"""
	var json_message = JSON.stringify(message)
	udp.set_dest_address(PYTHON_IP, PYTHON_UDP_PORT)
	var error = udp.put_packet(json_message.to_utf8_buffer())
	
	if error != OK:
		print("❌ Error enviando mensaje a Python: ", error)

# ==============================================================================
# API PÚBLICA OPTIMIZADA - SISTEMA DE REGISTRO
# ==============================================================================

func register_ui_system(ui_system):
	"""Registra un sistema UI para recibir actualizaciones de neurofeedback"""
	if not ui_system in _registered_ui_systems and is_instance_valid(ui_system):
		_registered_ui_systems.append(ui_system)
		print("✅ UI system registrado para neurofeedback")

func unregister_ui_system(ui_system):
	"""Elimina un sistema UI del registro"""
	if ui_system in _registered_ui_systems:
		_registered_ui_systems.erase(ui_system)

func register_minigame(minigame):
	"""Registra un minijuego para recibir actualizaciones de neurofeedback"""
	if not minigame in _registered_minigames and is_instance_valid(minigame):
		_registered_minigames.append(minigame)
		print("✅ Minijuego registrado para neurofeedback")

func unregister_minigame(minigame):
	"""Elimina un minijuego del registro"""
	if minigame in _registered_minigames:
		_registered_minigames.erase(minigame)

func get_current_ratio() -> float:
	"""Retorna el ratio actual de neurofeedback"""
	return current_ratio

func set_game_state(state: String):
	"""Establece estado del juego (para casos específicos)"""
	if state != current_game_state:
		current_game_state = state
		print("🎮 UDP - Estado del juego forzado: ", state)
		_force_send_game_state()

func set_energy(energy: int):
	"""Establece energía (para casos específicos)"""
	if energy != current_energy:
		current_energy = energy
		print("🔋 UDP - Energía forzada: ", energy)
		_force_send_game_state()

# NUEVO: Función para establecer el estado de la misión
func set_mission_state(state: String):
	"""Establece el estado de la misión desde el GameUI - VERSIÓN MEJORADA"""
	if state != current_mission_state:
		current_mission_state = state
		print("🎯 UDP - Estado de misión actualizado: ", state)
		_queue_game_state_send()

# Funciones de compatibilidad para minijuegos específicos
func register_corsi_game(game_node: Node):
	"""Registra juego Corsi (para compatibilidad)"""
	corsi_game = game_node
	register_minigame(game_node)

func register_nback_game(game_node: Node):
	"""Registra juego N-Back (para compatibilidad)"""
	nback_game = game_node
	register_minigame(game_node)

func _exit_tree():
	if udp != null:
		udp.close()
