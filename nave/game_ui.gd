extends CanvasLayer

# Referencias a nodos
@onready var message_label: Label = $MainContainer/BottomPanel/VBoxContainer/MessageLabel
@onready var instructions_label: Label = $MainContainer/BottomPanel/VBoxContainer/InstructionsLabel
@onready var neuro_bar: ProgressBar = $MainContainer/NeuroFeedbackPanel/NeuroBar
@onready var neuro_feedback_label: Label = $MainContainer/NeuroFeedbackPanel/NeuroFeedbackLabel
@onready var progress_bar: ProgressBar = $MainContainer/ProgressBar
@onready var energy_label: Label = $MainContainer/EnergyLabel
@onready var mission_label: Label = $MainContainer/BottomPanel/VBoxContainer/MissionLabel

# Variables de neurofeedback - MODIFICADO para usar valor real
var brain_ratio: float = 0.0  # Cambiado a 0.0 inicial
var target_brain_ratio: float = 0.0
var neuro_update_speed: float = 2.0

# Variables para la energía de la nave - OPTIMIZADO
var current_energy: int = 0
var max_energy: int = 600  # 6 módulos × 100 unidades
var modules_completed: int = 0
var total_modules: int = 6

# Estados de la misión
enum MissionState { CRITICAL, LOW, STABLE, GOOD, OPTIMAL, COMPLETE, GAME_OVER }
var current_mission_state: MissionState = MissionState.CRITICAL

# Cache para optimización - NUEVO
var last_energy_displayed: int = -1
var last_mission_state_displayed: MissionState = MissionState.CRITICAL
var update_cooldown: float = 0.0
const UI_UPDATE_INTERVAL: float = 0.2  # Actualizar UI cada 200ms

# Cache para estilos - NUEVO
var _cached_fill_style: StyleBoxFlat = null
var _cached_bg_style: StyleBoxFlat = null

# NUEVO: Variables para neurofeedback real
var _neuro_connected: bool = false
var _last_neuro_update: float = 0.0
const NEURO_UPDATE_INTERVAL: float = 0.1  # Actualizar neurofeedback cada 100ms

func _ready():
	# Configuración inicial
	setup_styles()
	set_default_messages()
	update_neuro_bar()
	update_energy_display()
	
	# Configurar la barra de energía principal
	progress_bar.min_value = 0
	progress_bar.max_value = max_energy
	progress_bar.value = current_energy
	
	# Conectar con sistema de neurofeedback real
	_connect_to_neurofeedback()
	
	# NUEVO: Conectar a señal de PlayerState
	_connect_to_player_state()
	
	# Iniciar con mensaje de contexto
	show_context_message()

func _connect_to_player_state():
	"""Conecta con PlayerState para recibir actualizaciones inmediatas"""
	if has_node("/root/PlayerState"):
		var player_state = get_node("/root/PlayerState")
		if player_state.has_signal("energy_updated"):
			player_state.energy_updated.connect(_on_player_energy_updated)
			print("✅ GameUI conectado a señales de PlayerState")
		
		# Forzar una actualización inicial
		update_energy_from_global()

func _on_player_energy_updated(total_energy: int, corsi_energy: int, nback_energy: int):
	"""Manejador de señal cuando la energía cambia en PlayerState"""
	print("🔔 GameUI - Señal de energía recibida: ", total_energy)
	
	# Actualizar inmediatamente
	current_energy = total_energy
	
	# Actualizar módulos completados
	if has_node("/root/PlayerState"):
		var player_state = get_node("/root/PlayerState")
		var progress = player_state.get_module_progress()
		modules_completed = progress.corsi_completed + progress.nback_completed
	
	update_energy_display()
	update_mission_state()
	update_mission_message()
	
	# Actualizar UDP
	_update_udp_mission_state()

func update_energy_from_global():
	"""Actualiza la energía desde el sistema global PlayerState - VERSIÓN MEJORADA"""
	if has_node("/root/PlayerState"):
		var player_state = get_node("/root/PlayerState")
		var new_energy = player_state.get_total_energy()
		var progress = player_state.get_module_progress()
		var new_total_modules = progress.corsi_completed + progress.nback_completed
		
		# DEBUG: Información detallada
		#print("🔍 GameUI - Verificando energía:")
		#print("   PlayerState reporta: ", new_energy)
		#print("   GameUI tiene: ", current_energy)
		#print("   Módulos completados: ", new_total_modules)
		#print("   Corsi: ", progress.corsi_energy, " | NBack: ", progress.nback_energy)
		
		# Solo actualizar si hay cambios - OPTIMIZACIÓN
		var energy_changed = new_energy != current_energy
		var modules_changed = new_total_modules != modules_completed
		
		if energy_changed or modules_changed:
			current_energy = new_energy
			modules_completed = new_total_modules
			
			print("🔄 GameUI - Actualizando energía: ", current_energy)
			
			# CORRECCIÓN: Actualizar el estado de la misión ANTES de mostrar la energía
			update_mission_state()
			update_energy_display()
			update_mission_message()
			
			# NUEVO: Actualizar inmediatamente el estado en UDP cuando cambia la energía
			_update_udp_mission_state()
			
			# Mostrar mensaje especial cuando se completa un módulo - OPTIMIZADO
			if modules_changed and modules_completed > 0 and modules_completed <= 6:
				show_priority_message("✅ ¡Módulo completado! Dirígete al siguiente objetivo", 3.0)
	
	# NUEVO: Forzar actualización del estado de misión incluso si no hay cambios de energía
	elif _neuro_connected:
		_update_udp_mission_state()

func _connect_to_neurofeedback():
	"""Conecta con el sistema de neurofeedback real"""
	if has_node("/root/NeuroFeedbackUDP"):
		var neuro_system = get_node("/root/NeuroFeedbackUDP")
		# Registrar este UI para recibir actualizaciones
		neuro_system.register_ui_system(self)
		_neuro_connected = true
		print("✅ GameUI conectado al sistema de neurofeedback real")
	else:
		print("⚠️ Sistema de neurofeedback no encontrado, usando valores de simulación")

func _process(delta):
	# Suavizar la actualización de la barra de neurofeedback
	if abs(neuro_bar.value - target_brain_ratio) > 0.01:
		neuro_bar.value = lerp(neuro_bar.value, target_brain_ratio, neuro_update_speed * delta)
		update_neuro_color()
	
	# Actualizar energía con control de frecuencia - OPTIMIZADO
	update_cooldown += delta
	if update_cooldown >= UI_UPDATE_INTERVAL:
		update_energy_from_global()
		update_cooldown = 0.0
	
	# NUEVO: Actualizar neurofeedback real periódicamente
	_last_neuro_update += delta
	if _last_neuro_update >= NEURO_UPDATE_INTERVAL:
		_update_real_neurofeedback()
		_last_neuro_update = 0.0

func _update_real_neurofeedback():
	"""Actualiza el neurofeedback desde el sistema real"""
	if _neuro_connected and has_node("/root/NeuroFeedbackUDP"):
		var neuro_system = get_node("/root/NeuroFeedbackUDP")
		var current_neuro = neuro_system.get_current_ratio()
		
		# Solo actualizar si hay un cambio significativo
		if abs(current_neuro - brain_ratio) > 0.01:
			set_brain_ratio(current_neuro)
	else:
		# Si no hay conexión, mostrar estado de conexión
		neuro_feedback_label.text = "NEUROFEEDBACK (SIN CONEXIÓN)"
		neuro_bar.value = 0.0

func setup_styles():
	# Configurar panel inferior con estilo moderno - MÁS TRANSPARENTE
	var bottom_style = StyleBoxFlat.new()
	bottom_style.bg_color = Color(0.02, 0.02, 0.05, 0.85)  # Más transparente
	bottom_style.border_color = Color(0.3, 0.6, 1.0, 0.8)
	bottom_style.border_width_left = 3
	bottom_style.border_width_right = 3
	bottom_style.border_width_top = 3
	bottom_style.border_width_bottom = 3
	bottom_style.corner_radius_top_left = 20
	bottom_style.corner_radius_top_right = 20
	bottom_style.corner_radius_bottom_left = 0
	bottom_style.corner_radius_bottom_right = 0
	bottom_style.shadow_color = Color(0, 0.3, 0.8, 0.3)
	bottom_style.shadow_size = 15
	$MainContainer/BottomPanel.add_theme_stylebox_override("panel", bottom_style)
	
	# Agregar esto en setup_styles() después de configurar bottom_style
	if not has_node("MainContainer/BottomPanel/Background"):
		var background = ColorRect.new()
		background.name = "Background"
		background.anchor_left = 0
		background.anchor_top = 0
		background.anchor_right = 1
		background.anchor_bottom = 1
		background.color = Color(0.02, 0.02, 0.05, 0.85)
		$MainContainer/BottomPanel.add_child(background)
		background.z_index = -1  # Para que esté detrás del texto
	
	# Configurar estilo para las etiquetas - TEXTO MÁS GRANDE
	message_label.add_theme_color_override("font_color", Color(0.95, 0.97, 1.0))
	message_label.add_theme_font_size_override("font_size", 26)  # Aumentado
	message_label.add_theme_constant_override("outline_size", 3)
	message_label.add_theme_color_override("font_outline_color", Color(0, 0.1, 0.3))
	
	instructions_label.add_theme_color_override("font_color", Color(0.8, 0.85, 1.0))
	instructions_label.add_theme_font_size_override("font_size", 18)  # Aumentado
	
	# Configurar etiqueta de neurofeedback - MODIFICADO para mostrar porcentaje
	neuro_feedback_label.add_theme_color_override("font_color", Color(0.9, 0.95, 1.0))
	neuro_feedback_label.add_theme_font_size_override("font_size", 16)
	neuro_feedback_label.text = "NF: 0%"  # Texto inicial
	
	# Configurar etiqueta de energía - SIMPLIFICADA
	energy_label.add_theme_color_override("font_color", Color(1.0, 0.9, 0.3))
	energy_label.add_theme_font_size_override("font_size", 18)
	energy_label.add_theme_constant_override("outline_size", 2)
	energy_label.add_theme_color_override("font_outline_color", Color(0.2, 0.1, 0.0))
	
	# Configurar etiqueta de misión - TEXTO MÁS GRANDE
	mission_label.add_theme_color_override("font_color", Color(0.9, 0.95, 1.0))
	mission_label.add_theme_font_size_override("font_size", 18)  # Aumentado
	mission_label.add_theme_constant_override("outline_size", 2)
	mission_label.add_theme_color_override("font_outline_color", Color(0, 0.1, 0.2))
	
	# Pre-crear estilos para la barra de energía - OPTIMIZACIÓN
	_cached_bg_style = StyleBoxFlat.new()
	_cached_bg_style.bg_color = Color(0.15, 0.15, 0.25, 0.9)
	_cached_bg_style.border_color = Color(0.4, 0.4, 0.6)
	_cached_bg_style.border_width_left = 2
	_cached_bg_style.border_width_right = 2
	_cached_bg_style.border_width_top = 2
	_cached_bg_style.border_width_bottom = 2
	_cached_bg_style.corner_radius_top_left = 8
	_cached_bg_style.corner_radius_top_right = 8
	_cached_bg_style.corner_radius_bottom_left = 8
	_cached_bg_style.corner_radius_bottom_right = 8
	
	_cached_fill_style = StyleBoxFlat.new()
	_cached_fill_style.border_width_left = 2
	_cached_fill_style.border_width_right = 2
	_cached_fill_style.border_width_top = 2
	_cached_fill_style.border_width_bottom = 2
	_cached_fill_style.corner_radius_top_left = 8
	_cached_fill_style.corner_radius_top_right = 8
	_cached_fill_style.corner_radius_bottom_left = 8
	_cached_fill_style.corner_radius_bottom_right = 8

func update_mission_state():
	"""Actualiza el estado de la misión basado en la energía actual - VERSIÓN MEJORADA"""
	var old_state = current_mission_state
	
	# CORRECCIÓN: Usar umbrales más apropiados para el rango de 0-600
	if current_energy >= 550:  # 91.6% - Casi completo
		current_mission_state = MissionState.COMPLETE
	elif current_energy >= 450:  # 75% - Óptimo
		current_mission_state = MissionState.OPTIMAL
	elif current_energy >= 350:  # 58.3% - Bueno
		current_mission_state = MissionState.GOOD
	elif current_energy >= 250:  # 41.6% - Estable
		current_mission_state = MissionState.STABLE
	elif current_energy >= 100:  # 16.6% - Bajo
		current_mission_state = MissionState.LOW
	else:  # Menos de 100 - Crítico
		current_mission_state = MissionState.CRITICAL
	
	# DEBUG: Mostrar el estado actual para verificación
	print("🎯 GameUI - Estado de misión: ", _get_mission_state_string(), " | Energía: ", current_energy)
	
	# Si el estado cambió, mostrar mensaje especial - OPTIMIZADO
	if old_state != current_mission_state:
		show_state_transition_message(old_state, current_mission_state)
		last_mission_state_displayed = current_mission_state
		# NUEVO: Actualizar UDP inmediatamente cuando cambia el estado
		_update_udp_mission_state()

func _update_udp_mission_state():
	"""Actualiza el estado de la misión en el sistema UDP - VERSIÓN MEJORADA"""
	if _neuro_connected and has_node("/root/NeuroFeedbackUDP"):
		var neuro_system = get_node("/root/NeuroFeedbackUDP")
		var mission_state_str = _get_mission_state_string()
		print("📡 GameUI - Enviando estado de misión a UDP: ", mission_state_str)
		neuro_system.set_mission_state(mission_state_str)

func _get_mission_state_string() -> String:
	"""Convierte el estado de misión a string para UDP - VERSIÓN MEJORADA"""
	match current_mission_state:
		MissionState.CRITICAL:
			return "critical"
		MissionState.LOW:
			return "low"
		MissionState.STABLE:
			return "stable"
		MissionState.GOOD:
			return "good"
		MissionState.OPTIMAL:
			return "optimal"
		MissionState.COMPLETE:
			return "complete"
		MissionState.GAME_OVER:
			return "game_over"
		_:
			return "unknown"

func show_context_message():
	"""Muestra el mensaje de contexto inicial"""
	message_label.text = "🚨 SISTEMA DE EMERGENCIA - ASTRA-9 🚨"
	mission_label.text = "🤖 UNIDAD R-17: La nave está varada. Reactiva los 6 módulos energéticos."
	
	await get_tree().create_timer(5.0).timeout
	set_default_messages()
	update_mission_message()

func set_default_messages():
	message_label.text = "🤖 UNIDAD R-17 - SISTEMA OPERATIVO"
	instructions_label.text = "🎮 Mov: W A S D | ✨ Saltar: Espacio | 📷 Cámara: C "

func update_energy_display():
	"""Actualiza la barra y etiqueta de energía - VERSIÓN CORREGIDA"""
	# Solo actualizar si hay cambios - OPTIMIZACIÓN
	if current_energy != last_energy_displayed:
		progress_bar.value = current_energy
		update_energy_bar_color()
		
		# Calcular módulos completados aproximados (cada 100 energía = 1 módulo)
		modules_completed = min(total_modules, int(current_energy / 100))
		
		# ACTUALIZACIÓN CORREGIDA - usar el estado actualizado
		var status_text = ""
		match current_mission_state:
			MissionState.CRITICAL:
				status_text = "🚨 SISTEMA CRÍTICO"
			MissionState.LOW:
				status_text = "🔴 ENERGÍA BAJA"
			MissionState.STABLE:
				status_text = "🟡 SISTEMA ESTABLE"
			MissionState.GOOD:
				status_text = "🟢 BUEN RENDIMIENTO"
			MissionState.OPTIMAL:
				status_text = "💚 EFICIENCIA ÓPTIMA"
			MissionState.COMPLETE:
				status_text = "✅ CARGA COMPLETA"
		
		energy_label.text = status_text + " (" + str(current_energy) + "/" + str(max_energy) + ")"
		last_energy_displayed = current_energy

func update_mission_message():
	"""Actualiza el mensaje de misión según el progreso - VERSIÓN MEJORADA"""
	var mission_text = ""
	
	# Obtener información de secuencia desde PlayerState
	var next_module_type = "explorar"
	if has_node("/root/PlayerState"):
		var player_state = get_node("/root/PlayerState")
		var progress = player_state.get_module_progress()
		next_module_type = progress.next_module_type
	
	# Mensajes basados en la secuencia esperada
	match next_module_type:
		"corsi":
			match modules_completed:
				0:
					mission_text = "🎯 OBJETIVO ACTUAL: Sala de Comando - Módulo CORSI (1)"
				2:
					mission_text = "🎯 OBJETIVO ACTUAL: Generador Auxiliar - Módulo CORSI (2)"
				4:
					mission_text = "🎯 OBJETIVO ACTUAL: Cápsula Criogénesis - Módulo CORSI (3)"
				_:
					mission_text = "🎯 BUSCAR MÓDULO CORSI - Consulta el mapa"
		
		"nback":
			match modules_completed:
				1:
					mission_text = "🎯 OBJETIVO ACTUAL: Sala de Reuniones - Módulo N-BACK (1)"
				3:
					mission_text = "🎯 OBJETIVO ACTUAL: Sala de Comando - Módulo N-BACK (2)"
				5:
					mission_text = "🎯 OBJETIVO ACTUAL: Dormitorios - Módulo N-BACK (3)"
				_:
					mission_text = "🎯 BUSCAR MÓDULO N-BACK - Consulta el mapa"
		
		"complete":
			mission_text = "🎉 ¡MISIÓN CUMPLIDA! Todos los módulos reactivados"
		
		_:
			mission_text = "🎯 EXPLORA LA NAVE PARA ENCONTRAR MÓDULOS"
	
	mission_label.text = mission_text

func show_state_transition_message(old_state: MissionState, new_state: MissionState):
	"""Muestra mensaje cuando cambia el estado de la misión - OPTIMIZADO"""
	var message = ""
	
	match new_state:
		MissionState.LOW:
			message = "⚡ ENERGÍA MÍNIMA DETECTADA"
		MissionState.STABLE:
			message = "✅ FUNCIONAMIENTO BÁSICO RESTAURADO"
		MissionState.GOOD:
			message = "🟢 SISTEMAS PRINCIPALES OPERATIVOS"
		MissionState.OPTIMAL:
			message = "💚 EFICIENCIA EN MÁXIMOS"
		MissionState.COMPLETE:
			message = "🎉 CARGA COMPLETA"
	
	if message:
		show_priority_message(message, 4.0)

func update_energy_bar_color():
	"""Actualiza el color de la barra de energía según el nivel - VERSIÓN OPTIMIZADA"""
	# Usar estilos pre-creados en lugar de crear nuevos - OPTIMIZACIÓN
	
	# Configurar color de relleno según estado de misión - COLORES MÁS VIVOS
	match current_mission_state:
		MissionState.CRITICAL:
			_cached_fill_style.bg_color = Color(0.9, 0.1, 0.1)  # Rojo más vivo
			# Efecto de parpadeo rojo de emergencia
			var pulse = sin(Time.get_ticks_msec() * 0.008) * 0.3 + 0.7
			_cached_fill_style.bg_color = Color(0.9, 0.1, 0.1) * pulse
			
		MissionState.LOW:
			_cached_fill_style.bg_color = Color(0.95, 0.5, 0.1)  # Naranja más vivo
			
		MissionState.STABLE:
			_cached_fill_style.bg_color = Color(0.95, 0.85, 0.2)  # Amarillo más vivo
			
		MissionState.GOOD:
			_cached_fill_style.bg_color = Color(0.3, 0.85, 0.2)  # Verde más vivo
			
		MissionState.OPTIMAL:
			_cached_fill_style.bg_color = Color(0.1, 0.95, 0.3)  # Verde brillante
			# Efecto de brillo verde
			var glow = sin(Time.get_ticks_msec() * 0.005) * 0.15 + 0.85
			_cached_fill_style.bg_color = Color(0.1, 0.95, 0.3) * glow
			
		MissionState.COMPLETE:
			_cached_fill_style.bg_color = Color(0.1, 0.9, 0.6)  # Verde azulado
			# Efecto de arcoíris para misión completada
			var rainbow = 0.5 + sin(Time.get_ticks_msec() * 0.003) * 0.3
			_cached_fill_style.bg_color = Color(0.1 + rainbow * 0.5, 0.9, 0.3 + rainbow * 0.3)
			_cached_fill_style.shadow_color = Color(0.1, 0.8, 0.5, 0.6)
			_cached_fill_style.shadow_size = 8
	
	# APLICAR LOS ESTILOS - REUTILIZANDO OBJETOS EXISTENTES
	progress_bar.add_theme_stylebox_override("background", _cached_bg_style)
	progress_bar.add_theme_stylebox_override("fill", _cached_fill_style)
	
	# Forzar actualización visual
	progress_bar.queue_redraw()

func show_priority_message(text: String, duration: float = 5.0):
	"""Muestra un mensaje prioritario con efectos especiales"""
	message_label.text = text
	
	# Efecto visual especial para mensajes importantes
	var tween = create_tween()
	tween.tween_property(message_label, "scale", Vector2(1.1, 1.1), 0.2)
	tween.tween_property(message_label, "scale", Vector2(1.0, 1.0), 0.2)
	tween.set_loops(2)
	
	# Cambiar color según el tipo de mensaje
	if "CRÍTICO" in text or "FALLO" in text:
		message_label.add_theme_color_override("font_color", Color(1, 0.3, 0.3))
	elif "COMPLETA" in text or "ÉXITO" in text:
		message_label.add_theme_color_override("font_color", Color(0.3, 1, 0.3))
	elif "ADVERTENCIA" in text:
		message_label.add_theme_color_override("font_color", Color(1, 0.8, 0.3))
	
	if duration > 0:
		await get_tree().create_timer(duration).timeout
		message_label.add_theme_color_override("font_color", Color(0.9, 0.95, 1.0))
		set_default_messages()

func show_message(text: String, duration: float = 5.0):
	message_label.text = text
	# Efecto visual para mensajes normales
	var tween = create_tween()
	tween.tween_property(message_label, "modulate", Color(1, 1, 1, 1), 0.2)
	tween.tween_property(message_label, "modulate", Color(1, 1, 1, 0.8), 0.1)
	tween.tween_property(message_label, "modulate", Color(1, 1, 1, 1), 0.1)
	
	if duration > 0:
		await get_tree().create_timer(duration).timeout
		set_default_messages()

func show_instructions(text: String):
	instructions_label.text = text

func update_neuro_bar():
	"""Actualiza la barra de neurofeedback - MODIFICADO para usar valor real"""
	neuro_bar.value = brain_ratio
	update_neuro_color()

func update_neuro_color():
	"""Actualiza el color de la barra de neurofeedback - MEJORADO"""
	var fill_style = StyleBoxFlat.new()
	var neuro_value = neuro_bar.value
	
	# NUEVO: Mostrar colores más precisos según el valor real
	if neuro_value < 0.3:
		fill_style.bg_color = Color(0.8, 0.2, 0.2)  # Rojo (bajo)
		neuro_feedback_label.add_theme_color_override("font_color", Color(1.0, 0.3, 0.3))
	elif neuro_value < 0.6:
		fill_style.bg_color = Color(0.9, 0.7, 0.2)  # Amarillo (medio)
		neuro_feedback_label.add_theme_color_override("font_color", Color(1.0, 0.8, 0.3))
	else:
		fill_style.bg_color = Color(0.2, 0.8, 0.4)  # Verde (alto)
		neuro_feedback_label.add_theme_color_override("font_color", Color(0.3, 1.0, 0.5))
	
	# Añadir efecto de brillo para valores altos
	if neuro_value > 0.8:
		fill_style.border_color = Color(1, 1, 1, 0.5)
		fill_style.border_width_left = 1
		fill_style.border_width_right = 1
		fill_style.border_width_top = 1
		fill_style.border_width_bottom = 1
		fill_style.shadow_color = Color(0.2, 0.8, 0.4, 0.3)
		fill_style.shadow_size = 4
	
	neuro_bar.add_theme_stylebox_override("fill", fill_style)

func set_brain_ratio(ratio: float):
	"""Establece ratio de neurofeedback optimizado - MODIFICADO para mostrar valor real"""
	target_brain_ratio = clamp(ratio, 0.0, 1.0)
	brain_ratio = target_brain_ratio
	
	# NUEVO: Actualizar etiqueta con valor numérico
	if _neuro_connected:
		var percentage = int(ratio * 100)
		neuro_feedback_label.text = "NF: %d%%" % percentage
	else:
		neuro_feedback_label.text = "NF"
	
	update_neuro_bar()

# Funciones para mostrar/ocultar elementos
func hide_neuro_feedback():
	$MainContainer/NeuroFeedbackPanel.visible = false

func show_neuro_feedback():
	$MainContainer/NeuroFeedbackPanel.visible = true

func hide_messages():
	$MainContainer/BottomPanel.visible = false

func show_messages():
	$MainContainer/BottomPanel.visible = true

# Efectos especiales para mensajes críticos
func show_alert_message(text: String, duration: float = 3.0):
	message_label.add_theme_color_override("font_color", Color(1, 0.3, 0.3))
	message_label.text = "🚨 " + text
	var tween = create_tween()
	tween.tween_property(message_label, "modulate", Color(1, 1, 1, 1), 0.1)
	tween.tween_property(message_label, "modulate", Color(1, 0.5, 0.5, 0.7), 0.1)
	tween.set_loops(4)
	
	if duration > 0:
		await get_tree().create_timer(duration).timeout
		message_label.add_theme_color_override("font_color", Color(0.9, 0.95, 1.0))
		set_default_messages()

func show_success_message(text: String, duration: float = 3.0):
	message_label.add_theme_color_override("font_color", Color(0.3, 1, 0.3))
	message_label.text = "✅ " + text
	
	if duration > 0:
		await get_tree().create_timer(duration).timeout
		message_label.add_theme_color_override("font_color", Color(0.9, 0.95, 1.0))
		set_default_messages()

# Función para mostrar mensaje de game over
func show_game_over():
	message_label.add_theme_color_override("font_color", Color(1, 0.2, 0.2))
	message_label.text = "💀 FALLO EN EL SISTEMA - GAME OVER"
	instructions_label.text = "La nave Astra-9 queda varada en el vacío espacial..."
	
	# Efecto de parpadeo rojo permanente
	var tween = create_tween()
	tween.tween_property(message_label, "modulate", Color(1, 0.3, 0.3, 0.5), 0.5)
	tween.tween_property(message_label, "modulate", Color(1, 0.2, 0.2, 1.0), 0.5)
	tween.set_loops()

# Función para mostrar mensaje de victoria
func show_victory():
	message_label.add_theme_color_override("font_color", Color(0.3, 1, 0.6))
	message_label.text = "🎉 CARGA COMPLETA - MISIÓN CUMPLIDA"
	instructions_label.text = "La nave Astra-9 está lista para regresar a su planeta de origen."
	
	# Efecto de brillo verde
	var tween = create_tween()
	tween.tween_property(message_label, "modulate", Color(1, 1, 1, 1), 0.3)
	tween.tween_property(message_label, "modulate", Color(0.6, 1, 0.8, 0.8), 0.3)
	tween.set_loops(3)

# Función para forzar actualización de energía (útil para debug)
func force_energy_update():
	update_energy_from_global()
