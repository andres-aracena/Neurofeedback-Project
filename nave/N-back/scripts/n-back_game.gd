extends Node

# ==============================================================================
# CONFIGURACIÓN DE CALIBRACIÓN (NUEVO SISTEMA)
# ==============================================================================
var calibration_buffer: Array = []
var is_calibrating: bool = false
var baseline_mean: float = 0.60 
var baseline_std: float = 0.05  

# [AJUSTE - DIFICULTAD ADAPTATIVA]
# Define qué tanto debe aumentar el ratio Theta/Gamma sobre el promedio
# para obtener bonificaciones.
# 0.2 = Fácil (Requiere poca carga extra de memoria)
# 0.6 = Estándar (Requiere esfuerzo cognitivo activo)
# 1.0 = Difícil (Requiere alto uso de memoria de trabajo)
const DIFFICULTY_FACTOR := 0.6 

var current_threshold: float = 0.80 

# ==============================================================================
# CONSTANTES DEL JUEGO
# ==============================================================================
const GRID_SIZE := 3

# [AJUSTE - VELOCIDAD DEL JUEGO]
# Tiempo que el estímulo permanece visible/audible
const STIMULUS_TIME := 0.5
# Tiempo de espera entre un estímulo y el siguiente (ventana de respuesta)
# Reducir esto (ej. 2.0) hace el juego más frenético.
const INTER_STIMULUS_TIME := 3.0

const TRIALS_PER_LEVEL := 7
const MAX_N_LEVEL := 3
const MAX_ENERGY := 100

# ==============================================================================
# SISTEMA DE ENERGÍA
# ==============================================================================
const BASE_ENERGY_CORRECT := 10
const BASE_ENERGY_WRONG := -5

# [AJUSTE - IMPACTO NEURO]
# Multiplicador de bonificación. 
# 2.0 = El estado mental puede duplicar la energía ganada.
const NEURO_BONUS_MULTIPLIER := 2.0

# Umbrales para la batería final
const CRITICAL_ENERGY := 30
const LOW_ENERGY := 50
const ADEQUATE_ENERGY := 70
const GOOD_ENERGY := 85
const EXCELLENT_ENERGY := 95

# ==============================================================================
# VARIABLES DE NODOS
# ==============================================================================
@onready var grid: GridContainer = $GridContainer
@onready var label_score: Label = $GUI/score
@onready var label_level: Label = $GUI/level
@onready var menu_label: Label = $Menu/Label
@onready var button: Button = $Menu/Button
@onready var menu: Control = $Menu
@onready var progress_bar: ProgressBar = $GUI/ProgressBar
@onready var feedback_label: Label = $GUI/feedback
@onready var neuro_bar: ProgressBar = $GUI/NeuroBar

# Nodos opcionales que se crean automáticamente
var audio_player: AudioStreamPlayer
var position_button: Button
var sound_button: Button

# ==============================================================================
# VARIABLES DEL JUEGO
# ==============================================================================
var energy_contribution: int = 0
var n_level: int = 1

# [VARIABLE BCI - THETA/GAMMA]
# Ratio de Memoria de Trabajo. Se actualiza vía UDP.
var brain_ratio: float = 0.8

var current_trial: int = 0
var correct_position_responses: int = 0
var correct_sound_responses: int = 0
var total_position_opportunities: int = 0
var total_sound_opportunities: int = 0

# Estadísticas
var perfect_neuro_count: int = 0
var total_neuro_sum: float = 0.0
var neuro_samples: int = 0
var response_accuracy: float = 0.0

var blocks: Array = []
var position_history: Array = []
var sound_history: Array = []
var current_position: int = -1
var current_sound: int = -1
var waiting_for_position_response: bool = false
var waiting_for_sound_response: bool = false

# Sistema de audio
var audio_samples: Array = []

# ==============================================================================
# ESTADOS
# ==============================================================================
enum NBackState {
	INTRO,
	CALIBRATING, # Nuevo estado
	SHOW_STIMULUS,
	WAITING_RESPONSE,
	FEEDBACK,
	RESULTS
}
var current_state: NBackState = NBackState.INTRO
var state_timer: float = 0.0

var _game_active: bool = false
var _neuro_track_timer: float = 0.0
const NEURO_TRACK_INTERVAL: float = 0.5

var _energy_saved: bool = false
var _game_completed: bool = false

# ==============================================================================
# CICLO DE VIDA
# ==============================================================================

func _ready() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	_create_missing_nodes()
	_setup_audio()
	_initialize_game()
	_apply_color_theme()
	
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").register_nback_game(self)
	
	print("N-Back: Module initialized") # Debug limpio

func _process(delta: float) -> void:
	state_timer += delta
	
	if _game_active:
		_handle_game_states()
		_update_ui()
		_track_neuro_performance(delta)

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		return_to_world()

# ==============================================================================
# SISTEMA DE CALIBRACIÓN (NUEVO)
# ==============================================================================

func start_calibration_phase():
	print("N-Back: Starting calibration")
	_update_game_state(NBackState.CALIBRATING)
	
	# Ocultar UI de juego para que no estorbe el mensaje
	progress_bar.hide()
	neuro_bar.hide()
	label_score.hide()
	label_level.hide()
	if position_button: position_button.hide()
	if sound_button: sound_button.hide()
	
	feedback_label.show()
	feedback_label.text = "CALIBRANDO SINCRONIZACIÓN...\n(Midiendo carga de memoria de trabajo... 5s)"
	feedback_label.add_theme_color_override("font_color", Color(1.0, 1.0, 0.0))
	
	calibration_buffer.clear()
	is_calibrating = true
	
	await get_tree().create_timer(5.0).timeout
	finish_calibration()

func finish_calibration():
	is_calibrating = false
	
	if calibration_buffer.size() < 10:
		print("N-Back: Warning - Low calibration data. Using defaults.")
		baseline_mean = 0.75
		baseline_std = 0.05
	else:
		var sum = 0.0
		for val in calibration_buffer: sum += val
		baseline_mean = sum / calibration_buffer.size()
		
		var sum_sq_diff = 0.0
		for val in calibration_buffer: sum_sq_diff += pow(val - baseline_mean, 2)
		baseline_std = sqrt(sum_sq_diff / calibration_buffer.size())
		baseline_std = max(baseline_std, 0.02)

	# Meta dinámica
	current_threshold = baseline_mean + (baseline_std * DIFFICULTY_FACTOR)
	current_threshold = clamp(current_threshold, 0.30, 0.92)
	
	print("N-Back: Calibration - Mean: %.2f, Std: %.2f, Target: %.2f" % [baseline_mean, baseline_std, current_threshold])
	
	feedback_label.text = "¡Sincronización Completada!"
	feedback_label.add_theme_color_override("font_color", Color(0.0, 1.0, 0.0))
	
	# Restaurar UI
	progress_bar.show()
	neuro_bar.show()
	label_score.show()
	label_level.show()
	if position_button: position_button.show()
	if sound_button: sound_button.show()
	
	await get_tree().create_timer(1.0).timeout
	
	# Iniciar el bucle de juego
	_generate_stimulus()
	_update_game_state(NBackState.SHOW_STIMULUS)
	state_timer = 0.0

# ==============================================================================
# LÓGICA DE AUDIO
# ==============================================================================

func _setup_audio() -> void:
	print("N-Back: Loading audio streams...")
	var sound_files = ["alphabet-a.mp3", "alphabet-e.mp3", "alphabet-i.mp3", "alphabet-o.mp3", "alphabet-u.mp3"]
	for file in sound_files:
		var stream = load("res://N-back/sounds/" + file)
		if stream: audio_samples.append(stream)
	print("N-Back: %d audio streams loaded" % audio_samples.size())

func _play_sound(sound_index: int) -> void:
	if audio_player and sound_index >= 0 and sound_index < audio_samples.size():
		audio_player.stream = audio_samples[sound_index]
		audio_player.play()

# ==============================================================================
# SISTEMA DE ENERGÍA ADAPTATIVO
# ==============================================================================

func calculate_neuro_bonus_adaptive(current_neuro: float) -> float:
	# Umbral dinámico
	var target = current_threshold
	var floor_val = baseline_mean - baseline_std
	
	if current_neuro < floor_val: return 0.0
	
	var denominator = target - floor_val
	if denominator <= 0.0001: denominator = 0.1
	
	var progress = (current_neuro - floor_val) / denominator
	progress = clamp(progress, 0.0, 1.0)
	
	return pow(progress, 1.5) * NEURO_BONUS_MULTIPLIER

func calculate_energy_contribution(is_correct: bool) -> int:
	var base_energy = BASE_ENERGY_CORRECT if is_correct else BASE_ENERGY_WRONG
	
	if not is_correct:
		return base_energy
	
	var neuro_bonus = calculate_neuro_bonus_adaptive(brain_ratio)
	var total_energy = base_energy + (base_energy * neuro_bonus)
	
	if brain_ratio >= current_threshold:
		total_energy += 3
		perfect_neuro_count += 1
	
	return int(max(total_energy, 1))

func save_energy_to_global() -> void:
	print("N-Back: Saving energy: ", energy_contribution)
	if has_node("/root/PlayerState"):
		get_node("/root/PlayerState").add_nback_energy(energy_contribution)
	else:
		print("N-Back: Error - PlayerState not found")

# ==============================================================================
# INICIALIZACIÓN
# ==============================================================================

func _initialize_game() -> void:
	_reset_ui()
	_create_blocks()
	_setup_buttons()
	_reset_performance_tracking()

func _reset_performance_tracking() -> void:
	energy_contribution = 0
	perfect_neuro_count = 0
	total_neuro_sum = 0.0
	neuro_samples = 0
	correct_position_responses = 0
	correct_sound_responses = 0
	total_position_opportunities = 0
	total_sound_opportunities = 0
	response_accuracy = 0.0
	_energy_saved = false
	_game_completed = false

func _apply_color_theme() -> void:
	var text_color: Color = Color(0.7, 0.9, 1.0)
	var intro_color: Color = Color(0.4, 0.7, 1.0)
	var energy_color: Color = Color(0.0, 0.8, 1.0)
	
	label_score.add_theme_color_override("font_color", energy_color)
	label_level.add_theme_color_override("font_color", text_color)
	feedback_label.add_theme_color_override("font_color", energy_color)
	menu_label.add_theme_color_override("font_color", intro_color)
	menu_label.add_theme_font_size_override("font_size", 24)
	
	if position_button:
		position_button.add_theme_color_override("font_color", text_color)
		position_button.add_theme_color_override("font_hover_color", energy_color)
	if sound_button:
		sound_button.add_theme_color_override("font_color", text_color)
		sound_button.add_theme_color_override("font_hover_color", energy_color)

func _create_missing_nodes() -> void:
	# Creación automática de nodos si faltan (sin cambios funcionales, solo debug limpio)
	if has_node("AudioStreamPlayer"): audio_player = $AudioStreamPlayer
	else:
		audio_player = AudioStreamPlayer.new()
		audio_player.name = "AudioStreamPlayer"
		add_child(audio_player)
	
	if has_node("GUI/PositionButton"): position_button = $GUI/PositionButton
	else:
		position_button = Button.new()
		position_button.name = "PositionButton"
		position_button.text = "COINCIDENCIA POSICIÓN"
		position_button.custom_minimum_size = Vector2(240, 50)
		position_button.position = Vector2(250, 530)
		var style = StyleBoxFlat.new()
		style.bg_color = Color(0.2, 0.5, 0.8)
		position_button.add_theme_stylebox_override("normal", style)
		$GUI.add_child(position_button)
	
	if has_node("GUI/SoundButton"): sound_button = $GUI/SoundButton
	else:
		sound_button = Button.new()
		sound_button.name = "SoundButton"
		sound_button.text = "COINCIDENCIA SONIDO"
		sound_button.custom_minimum_size = Vector2(240, 50)
		sound_button.position = Vector2(680, 530)
		var style = StyleBoxFlat.new()
		style.bg_color = Color(0.8, 0.5, 0.2)
		sound_button.add_theme_stylebox_override("normal", style)
		$GUI.add_child(sound_button)

func _setup_buttons() -> void:
	if position_button:
		if position_button.pressed.is_connected(_on_position_button_pressed):
			position_button.pressed.disconnect(_on_position_button_pressed)
		position_button.pressed.connect(_on_position_button_pressed)
	
	if sound_button:
		if sound_button.pressed.is_connected(_on_sound_button_pressed):
			sound_button.pressed.disconnect(_on_sound_button_pressed)
		sound_button.pressed.connect(_on_sound_button_pressed)

func _create_blocks() -> void:
	for child in grid.get_children(): child.queue_free()
	blocks.clear()
	grid.columns = GRID_SIZE
	var tile_scene: PackedScene = preload("res://N-back/scenes/N-backTile.tscn")
	for i in range(GRID_SIZE * GRID_SIZE):
		var block = tile_scene.instantiate()
		grid.add_child(block)
		blocks.append(block)
		block.setup(i / GRID_SIZE, i % GRID_SIZE, i)

# ==============================================================================
# UI
# ==============================================================================

func _reset_ui() -> void:
	label_score.hide()
	label_level.hide()
	progress_bar.hide()
	feedback_label.hide()
	neuro_bar.hide()
	if position_button: position_button.hide()
	if sound_button: sound_button.hide()
	
	button.show()
	menu_label.show()
	menu.show()
	
	# TEXTO MODIFICADO: Contexto + Instrucciones
	menu_label.text = "🔋 MÓDULO N-BACK: SINCRONIZACIÓN\n\n" + \
					  "El núcleo de datos pierde coherencia. Usa tu memoria de trabajo para sincronizarlo.\n\n" + \
					  "CÓMO JUGAR:\n" + \
					  "Se presentará una secuencia de posiciones y sonidos.\n" + \
					  "• Presiona 'COINCIDENCIA POSICIÓN' si el lugar es igual al de hace N turnos.\n" + \
					  "• Presiona 'COINCIDENCIA SONIDO' si la letra es igual a la de hace N turnos.\n\n" + \
					  "El sistema se auto-calibrará al iniciar."

func _update_ui() -> void:
	if label_score: label_score.text = "%d energía" % energy_contribution
	if label_level: label_level.text = "Nivel %d-back" % n_level
	if progress_bar: progress_bar.value = float(current_trial) / TRIALS_PER_LEVEL
	if neuro_bar: neuro_bar.value = brain_ratio

# ==============================================================================
# LÓGICA DE JUEGO
# ==============================================================================

func _handle_game_states() -> void:
	match current_state:
		NBackState.CALIBRATING: pass
		NBackState.SHOW_STIMULUS:
			if state_timer >= STIMULUS_TIME:
				_hide_current_stimulus()
				_update_game_state(NBackState.WAITING_RESPONSE)
				state_timer = 0.0
				waiting_for_position_response = _has_position_match()
				waiting_for_sound_response = _has_sound_match()
				feedback_label.text = "🎯 Detecta patrones..."
		NBackState.WAITING_RESPONSE:
			if state_timer >= INTER_STIMULUS_TIME:
				if waiting_for_position_response: _register_position_response(false)
				if waiting_for_sound_response: _register_sound_response(false)
				_generate_next_stimulus()
		NBackState.FEEDBACK:
			if state_timer >= 0.8: _generate_next_stimulus()

func _generate_next_stimulus() -> void:
	current_trial += 1
	if current_trial > TRIALS_PER_LEVEL: _finish_level()
	else:
		_generate_stimulus()
		_update_game_state(NBackState.SHOW_STIMULUS)
		state_timer = 0.0

func _generate_stimulus() -> void:
	var position_match = false
	if position_history.size() >= n_level: position_match = randf() < 0.3
	
	var sound_match = false
	if sound_history.size() >= n_level: sound_match = randf() < 0.3
	
	if position_match and position_history.size() >= n_level:
		current_position = position_history[position_history.size() - n_level]
	else:
		var new_position
		while true:
			new_position = randi() % (GRID_SIZE * GRID_SIZE)
			if position_history.size() == 0 or new_position != position_history[-1]: break
		current_position = new_position
	
	if sound_match and sound_history.size() >= n_level:
		current_sound = sound_history[sound_history.size() - n_level]
	else:
		var new_sound
		while true:
			new_sound = randi() % audio_samples.size()
			if sound_history.size() == 0 or new_sound != sound_history[-1]: break
		current_sound = new_sound
	
	position_history.append(current_position)
	sound_history.append(current_sound)
	_show_current_stimulus()

func _show_current_stimulus() -> void:
	for block in blocks: block.set_highlighted(false)
	if current_position >= 0 and current_position < blocks.size():
		blocks[current_position].set_highlighted(true)
	_play_sound(current_sound)
	if feedback_label:
		feedback_label.text = " Transmisión %d/%d - Nivel %d-back" % [current_trial, TRIALS_PER_LEVEL, n_level]

func _hide_current_stimulus() -> void:
	for block in blocks: block.set_highlighted(false)

# ==============================================================================
# COINCIDENCIAS Y RESPUESTAS
# ==============================================================================

func _has_position_match() -> bool:
	if position_history.size() <= n_level: return false
	return current_position == position_history[position_history.size() - n_level - 1]

func _has_sound_match() -> bool:
	if sound_history.size() <= n_level: return false
	return current_sound == sound_history[sound_history.size() - n_level - 1]

func _on_position_button_pressed() -> void:
	if current_state == NBackState.WAITING_RESPONSE:
		_register_position_response(waiting_for_position_response)

func _on_sound_button_pressed() -> void:
	if current_state == NBackState.WAITING_RESPONSE:
		_register_sound_response(waiting_for_sound_response)

func _register_position_response(correct: bool) -> void:
	if waiting_for_position_response:
		waiting_for_position_response = false
		total_position_opportunities += 1
		var energy_earned = calculate_energy_contribution(correct)
		energy_contribution = max(0, energy_contribution + energy_earned)
		
		if correct:
			correct_position_responses += 1
			var neuro_bonus = calculate_neuro_bonus_adaptive(brain_ratio)
			var msg = "✅ Posición correcta! +%d energía" % energy_earned
			if neuro_bonus > 0: msg += " | Eficiencia: +%.0f%%" % [neuro_bonus * 100]
			feedback_label.text = msg
			feedback_label.add_theme_color_override("font_color", Color(0.0, 0.8, 0.6))
		else:
			feedback_label.text = "❌ Error posición -%d energía" % abs(energy_earned)
			feedback_label.add_theme_color_override("font_color", Color(0.9, 0.3, 0.3))
		
		if has_node("/root/NeuroFeedbackUDP"):
			get_node("/root/NeuroFeedbackUDP").send_module_event("nback", "position_response", correct)
		
		_update_game_state(NBackState.FEEDBACK)
		state_timer = 0.0

func _register_sound_response(correct: bool) -> void:
	if waiting_for_sound_response:
		waiting_for_sound_response = false
		total_sound_opportunities += 1
		var energy_earned = calculate_energy_contribution(correct)
		energy_contribution = max(0, energy_contribution + energy_earned)

		if correct:
			correct_sound_responses += 1
			var neuro_bonus = calculate_neuro_bonus_adaptive(brain_ratio)
			var msg = "✅ Sonido correcto! +%d energía" % energy_earned
			if neuro_bonus > 0: msg += " | Eficiencia: +%.0f%%" % [neuro_bonus * 100]
			feedback_label.text = msg
			feedback_label.add_theme_color_override("font_color", Color(0.0, 0.8, 0.6))
		else:
			feedback_label.text = "❌ Error sonido -%d energía" % abs(energy_earned)
			feedback_label.add_theme_color_override("font_color", Color(0.9, 0.3, 0.3))
		
		if has_node("/root/NeuroFeedbackUDP"):
			get_node("/root/NeuroFeedbackUDP").send_module_event("nback", "sound_response", correct)
		
		_update_game_state(NBackState.FEEDBACK)
		state_timer = 0.0

func _track_neuro_performance(delta: float) -> void:
	if is_calibrating: return
	_neuro_track_timer += delta
	if _neuro_track_timer >= NEURO_TRACK_INTERVAL:
		total_neuro_sum += brain_ratio
		neuro_samples += 1
		_neuro_track_timer = 0.0

# ==============================================================================
# PROGRESIÓN
# ==============================================================================

func _finish_level() -> void:
	print("N-Back: Level %d completed" % n_level)
	if n_level < MAX_N_LEVEL:
		n_level += 1
		_play_level_up_effect()
		if feedback_label:
			feedback_label.add_theme_color_override("font_color", Color(0.0, 0.8, 0.6))
			feedback_label.text = "🔼 Avanzando a nivel " + str(n_level) + "-back"
	else:
		_show_final_results()
		return
	
	_start_next_level()

func _play_level_up_effect() -> void:
	label_level.pivot_offset = label_level.size / 2
	var tween = create_tween()
	tween.tween_property(label_level, "scale", Vector2(1.5, 1.5), 0.2).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tween.parallel().tween_property(label_level, "modulate", Color(1.0, 0.8, 0.2), 0.2) 
	tween.tween_interval(0.5)
	tween.tween_property(label_level, "scale", Vector2(1.0, 1.0), 0.5).set_trans(Tween.TRANS_ELASTIC).set_ease(Tween.EASE_OUT)
	tween.parallel().tween_property(label_level, "modulate", Color(0.7, 0.9, 1.0), 0.5)

func _start_next_level() -> void:
	current_trial = 0
	correct_position_responses = 0
	correct_sound_responses = 0
	total_position_opportunities = 0
	total_sound_opportunities = 0
	position_history.clear()
	sound_history.clear()
	current_state = NBackState.FEEDBACK
	state_timer = 0.0

# ==============================================================================
# CONTROL
# ==============================================================================

func start_game() -> void:
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").set_game_state("in_nback_minigame")
	
	_reset_performance_tracking()
	_game_active = true
	n_level = 1
	current_trial = 0
	_energy_saved = false
	_game_completed = false
	
	label_score.show()
	label_level.show()
	
	# Ocultamos botones de juego hasta calibrar
	if position_button: position_button.hide()
	if sound_button: sound_button.hide()
	
	button.hide()
	menu_label.hide()
	menu.hide()
	
	start_calibration_phase()

func _on_button_pressed() -> void:
	start_game()

func _show_final_results() -> void:
	label_score.hide()
	label_level.hide()
	progress_bar.hide()
	feedback_label.hide()
	neuro_bar.hide()
	if position_button: position_button.hide()
	if sound_button: sound_button.hide()
	
	button.hide()
	menu_label.show()
	menu.show()
	
	if not _energy_saved:
		save_energy_to_global()
		_energy_saved = true
		_game_completed = true
	
	var perf = get_detailed_performance()
	var assess = get_battery_assessment()
	
	var txt = "🚀 INFORME FINAL - MÓDULO N-BACK 🚀\n\n"
	txt += "⚡ ENERGÍA: %d/%d\n" % [energy_contribution, MAX_ENERGY]
	txt += "📊 Estado: %s\n\n" % assess.status
	txt += "🎯 ANÁLISIS:\n"
	txt += "• Precisión Global: %.1f%%\n" % (perf.response_accuracy * 100)
	txt += "• Memoria Trabajo (Theta/Gamma): %.1f%%\n\n" % perf.neuro_efficiency
	txt += "💡 IMPLICACIONES: " + assess.implication + "\n"
	txt += "\n🎮 ESC para volver"
	
	menu_label.text = txt
	_update_game_state(NBackState.RESULTS)

func return_to_world() -> void:
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").set_game_state("exploring")
		
	print("N-Back: Returning to world")
	if _game_completed and not _energy_saved:
		save_energy_to_global()
		_energy_saved = true
	
	get_tree().change_scene_to_file("res://main.tscn")
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func set_brain_ratio(ratio: float) -> void:
	brain_ratio = clamp(ratio, 0.0, 1.0)
	if is_calibrating:
		calibration_buffer.append(ratio)

func _update_game_state(new_state: NBackState):
	if current_state != new_state:
		current_state = new_state
		var state_str = ""
		match new_state:
			NBackState.INTRO: state_str = "INTRO"
			NBackState.CALIBRATING: state_str = "CALIBRATING"
			NBackState.SHOW_STIMULUS: state_str = "SHOW_STIMULUS"
			NBackState.WAITING_RESPONSE: state_str = "WAITING_RESPONSE"
			NBackState.FEEDBACK: state_str = "FEEDBACK"
			NBackState.RESULTS: state_str = "RESULTS"
		
		if has_node("/root/NeuroFeedbackUDP"):
			get_node("/root/NeuroFeedbackUDP").send_minigame_state("nback", state_str)

func get_detailed_performance() -> Dictionary:
	var avg_neuro = total_neuro_sum / neuro_samples if neuro_samples > 0 else 0.0
	var pos_acc = 0.0
	var snd_acc = 0.0
	if total_position_opportunities > 0: pos_acc = float(correct_position_responses) / total_position_opportunities
	if total_sound_opportunities > 0: snd_acc = float(correct_sound_responses) / total_sound_opportunities
	response_accuracy = (pos_acc + snd_acc) / 2.0 if (total_position_opportunities + total_sound_opportunities) > 0 else 0.0
	
	return {
		"energy_contribution": energy_contribution,
		"neuro_efficiency": avg_neuro * 100,
		"response_accuracy": response_accuracy
	}

func get_battery_assessment() -> Dictionary:
	var energy = energy_contribution
	var status = ""
	var implication = ""
	if energy <= CRITICAL_ENERGY:
		status = "CRÍTICO"
		implication = "Fallo de comunicaciones."
	elif energy <= LOW_ENERGY:
		status = "BAJO"
		implication = "Interferencias graves."
	elif energy <= ADEQUATE_ENERGY:
		status = "ADECUADO"
		implication = "Comunicaciones operativas."
	elif energy <= GOOD_ENERGY:
		status = "BUENO"
		implication = "Alta eficiencia."
	else:
		status = "EXCELENTE"
		implication = "Sincronización perfecta."
	return {"status": status, "implication": implication}
