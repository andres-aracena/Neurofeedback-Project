extends Node

# Energía acumulada - CORREGIDO
var corsi_energy: int = 0
var nback_energy: int = 0
var total_energy: int = 0

# Contadores de minijuegos
var corsi_modules_completed: int = 0
var nback_modules_completed: int = 0

# Control de secuencia - MODIFICADO: Más flexible
var expected_module_sequence: Array = ["corsi", "nback", "corsi", "nback", "corsi", "nback"]
var current_module_index: int = 0

# Umbrales de energía
const CRITICAL_TOTAL_ENERGY := 180
const LOW_TOTAL_ENERGY := 300
const ADEQUATE_TOTAL_ENERGY := 420
const GOOD_TOTAL_ENERGY := 500
const EXCELLENT_TOTAL_ENERGY := 550

# Cache para optimización
var last_energy_state: String = ""

# NUEVO: Señal para notificar cambios de energía
signal energy_updated(total_energy: int, corsi_energy: int, nback_energy: int)

func _ready():
	print("🔋 PlayerState inicializado")
	print("📋 Secuencia esperada: ", expected_module_sequence)

func add_corsi_energy(energy: int) -> void:
	"""Añade energía de Corsi - VERSIÓN MEJORADA"""
	print("🎯 Intentando añadir energía CORSI: ", energy)
	
	# VERIFICACIÓN: ¿Estamos en la secuencia correcta?
	var expected_module = get_current_expected_module()
	print("📋 Módulo esperado: ", expected_module, " | Índice actual: ", current_module_index)
	
	# NUEVO: Lógica mejorada de secuencia
	var should_advance_sequence = false
	
	if current_module_index < expected_module_sequence.size() and expected_module_sequence[current_module_index] == "corsi":
		should_advance_sequence = true
		print("✅ CORSI - Secuencia correcta, avanzando índice")
	else:
		print("⚠️ CORSI - Secuencia incorrecta. Esperaba: ", expected_module, " pero sumando energía de todos modos")
	
	# CORRECCIÓN: SUMAR energía siempre
	var old_corsi = corsi_energy
	corsi_energy += energy
	corsi_modules_completed += 1
	
	# Solo avanzar en la secuencia si es correcto
	if should_advance_sequence:
		current_module_index += 1
	
	print("📊 CORSI - Antes: ", old_corsi, " | Después: ", corsi_energy)
	print("📋 Nuevo índice de secuencia: ", current_module_index)
	
	_update_total_energy()
	_notify_energy_change()

func add_nback_energy(energy: int) -> void:
	"""Añade energía de N-Back - VERSIÓN MEJORADA"""
	print("🎯 Intentando añadir energía N-BACK: ", energy)
	
	# VERIFICACIÓN: ¿Estamos en la secuencia correcta?
	var expected_module = get_current_expected_module()
	print("📋 Módulo esperado: ", expected_module, " | Índice actual: ", current_module_index)
	
	# NUEVO: Lógica mejorada de secuencia
	var should_advance_sequence = false
	
	if current_module_index < expected_module_sequence.size() and expected_module_sequence[current_module_index] == "nback":
		should_advance_sequence = true
		print("✅ N-BACK - Secuencia correcta, avanzando índice")
	else:
		print("⚠️ N-BACK - Secuencia incorrecta. Esperaba: ", expected_module, " pero sumando energía de todos modos")
	
	# CORRECCIÓN: SUMAR energía siempre
	var old_nback = nback_energy
	nback_energy += energy
	nback_modules_completed += 1
	
	# Solo avanzar en la secuencia si es correcto
	if should_advance_sequence:
		current_module_index += 1
	
	print("📊 N-BACK - Antes: ", old_nback, " | Después: ", nback_energy)
	print("📋 Nuevo índice de secuencia: ", current_module_index)
	
	_update_total_energy()
	_notify_energy_change()
	
func _update_total_energy() -> void:
	"""Actualiza energía total con verificación de cambios - VERSIÓN MEJORADA"""
	var old_total = total_energy
	var new_total = corsi_energy + nback_energy
	
	if new_total != total_energy:
		total_energy = new_total
		print("⚡ ENERGÍA TOTAL ACTUALIZADA: ", old_total, " → ", total_energy)
		print("📊 Desglose - Corsi: ", corsi_energy, " | N-Back: ", nback_energy)
		
		# Emitir señal de cambio
		energy_updated.emit(total_energy, corsi_energy, nback_energy)

func _notify_energy_change():
	"""Notifica cambio de energía a sistemas externos"""
	print("🔔 Notificando cambio de energía: ", total_energy)
	# Esta función puede ser usada para notificar a la UI u otros sistemas

func get_total_energy() -> int:
	return total_energy

func get_current_expected_module() -> String:
	"""Obtiene el tipo de módulo esperado actualmente"""
	if current_module_index < expected_module_sequence.size():
		return expected_module_sequence[current_module_index]
	return "complete"

func get_module_progress() -> Dictionary:
	"""Obtiene el progreso actual de módulos"""
	return {
		"current_index": current_module_index,
		"total_modules": expected_module_sequence.size(),
		"next_module_type": get_current_expected_module(),
		"corsi_completed": corsi_modules_completed,
		"nback_completed": nback_modules_completed,
		"corsi_energy": corsi_energy,
		"nback_energy": nback_energy,
		"total_energy": total_energy
	}

func get_energy_status() -> Dictionary:
	"""Obtiene estado de energía con cache"""
	var total = get_total_energy()
	
	# Usar cache si es posible
	var current_state = _calculate_energy_status(total)
	if current_state.status != last_energy_state:
		last_energy_state = current_state.status
		print("🎯 Estado de energía: ", current_state.status)
	
	return current_state

func _calculate_energy_status(total: int) -> Dictionary:
	"""Calcula el estado de energía"""
	var status = ""
	var message = ""
	var color = Color.WHITE
	
	if total < CRITICAL_TOTAL_ENERGY:
		status = "CRÍTICO"
		message = "ENERGÍA INSUFICIENTE - SISTEMAS EN FALLO INMINENTE"
		color = Color(1.0, 0.2, 0.2)
	elif total < LOW_TOTAL_ENERGY:
		status = "BAJO"
		message = "ENERGÍA MÍNIMA - SISTEMAS OPERANDO EN EMERGENCIA"
		color = Color(1.0, 0.6, 0.2)
	elif total < ADEQUATE_TOTAL_ENERGY:
		status = "ESTABLE"
		message = "ENERGÍA ADECUADA - SISTEMAS PRINCIPALES OPERATIVOS"
		color = Color(1.0, 0.8, 0.2)
	elif total < GOOD_TOTAL_ENERGY:
		status = "BUENO"
		message = "BUENA ENERGÍA - SISTEMAS OPERANDO CON EFICIENCIA"
		color = Color(0.6, 0.8, 0.2)
	else:
		status = "ÓPTIMO"
		message = "ENERGÍA MÁXIMA - MISIÓN CUMPLIDA"
		color = Color(0.2, 0.8, 0.2)
	
	return {
		"status": status,
		"message": message,
		"color": color,
		"total_energy": total,
		"max_possible": 600,
		"progress_percentage": float(total) / 600.0 * 100.0
	}

func is_game_over() -> bool:
	return get_total_energy() < CRITICAL_TOTAL_ENERGY

func has_sufficient_energy() -> bool:
	return get_total_energy() >= ADEQUATE_TOTAL_ENERGY

func get_completion_percentage() -> float:
	return float(get_total_energy()) / 600.0 * 100.0

# NUEVO: Función para debug
func print_debug_info():
	"""Imprime información de debug del estado del jugador"""
	print("=== DEBUG PLAYERSTATE ===")
	print("🔋 Energía Total: ", total_energy)
	print("🎯 Corsi Energy: ", corsi_energy, " | Módulos: ", corsi_modules_completed)
	print("🎯 NBack Energy: ", nback_energy, " | Módulos: ", nback_modules_completed)
	print("📋 Secuencia - Índice: ", current_module_index, " | Esperado: ", get_current_expected_module())
	print("========================")

# Datos del jugador para guardar
var player_data: Dictionary = {
	"position": Vector3.ZERO,
	"rotation": Vector3.ZERO,
	"camera_mode": true,
	"camera_position": Vector3(0, 0.5, 0)
}

func save_player_state(player_node: Node) -> void:
	"""Guarda estado del jugador optimizado"""
	if player_node:
		player_data["position"] = player_node.global_position
		player_data["rotation"] = player_node.rotation
		player_data["camera_mode"] = player_node.first_person_mode
		player_data["camera_position"] = player_node.first_person_position

func load_player_state(player_node: Node) -> void:
	"""Carga estado del jugador optimizado"""
	if player_node and player_data["position"] != Vector3.ZERO:
		player_node.global_position = player_data["position"]
		player_node.rotation = player_data["rotation"]
		player_node.first_person_mode = player_data["camera_mode"]
		player_node.first_person_position = player_data["camera_position"]
		if player_node.has_method("update_camera_mode"):
			player_node.update_camera_mode()
