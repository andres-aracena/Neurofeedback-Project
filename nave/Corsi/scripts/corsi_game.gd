extends Node

# ==============================================================================
# CONFIGURACIÓN DE CALIBRACIÓN (NUEVO SISTEMA)
# ==============================================================================
var calibration_buffer: Array = []
var is_calibrating: bool = false
var baseline_mean: float = 0.60 
var baseline_std: float = 0.05  

# [AJUSTE IMPORTANTE - DIFICULTAD ADAPTATIVA]
# Define qué tanto esfuerzo extra (Z-Score) se requiere sobre el promedio base.
# 0.2 = Fácil (Requiere poca carga adicional de memoria de trabajo)
# 0.6 = Estándar (Requiere esfuerzo cognitivo activo)
# 1.0 = Difícil (Requiere alto uso de memoria de trabajo Theta/Gamma)
const DIFFICULTY_FACTOR := 0.6 

# Se calculará automáticamente. Es el valor objetivo de Theta/Gamma a superar.
var current_threshold: float = 0.80 

# ==============================================================================
# CONSTANTES DEL JUEGO
# ==============================================================================
const GRID_SIZE := 3

# [AJUSTE - VELOCIDAD]
# Tiempo en segundos que un bloque permanece encendido.
# Reducir este valor (ej. 0.4) aumenta drásticamente la dificultad de memoria.
const SHOW_TIME := 0.6
const GAP_TIME := 0.25

const GLOW_DURATION := 0.35
const START_DELAY := 1.0
const MAX_LEVEL := 5
const MAX_ENERGY := 100

# ==============================================================================
# SISTEMA DE ENERGÍA
# ==============================================================================
const BASE_ENERGY_CORRECT := 12
const BASE_ENERGY_WRONG := -6

# [AJUSTE - RECOMPENSA NEURO]
# Multiplicador de bonificación. Si es 2.0, un estado mental perfecto
# puede duplicar la energía ganada en ese turno.
const NEURO_BONUS_MULTIPLIER := 2.0

# Umbrales para la batería final
const CRITICAL_ENERGY := 30
const LOW_ENERGY := 50
const ADEQUATE_ENERGY := 70
const GOOD_ENERGY := 85
const EXCELLENT_ENERGY := 95

# ==============================================================================
# VARIABLES DE REFERENCIA A NODOS
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

# ==============================================================================
# VARIABLES DEL JUEGO
# ==============================================================================
var energy_contribution: int = 0
var level: int = 1

# [VARIABLE DE ENTRADA BCI]
# Representa el ratio Theta/Gamma. 
# Alto = Alta carga en Memoria de Trabajo (Esfuerzo mental).
# Bajo = Baja carga o distracción.
var brain_ratio: float = 0.8

# Estadísticas
var perfect_neuro_count: int = 0
var total_neuro_sum: float = 0.0
var neuro_samples: int = 0
var correct_sequences: int = 0
var total_sequences: int = 0
var sequence_accuracy: float = 0.0

var blocks: Array = []
var sequence: Array = []
var user_sequence: Array = []
var glow_timers: Array = []

# ==============================================================================
# ESTADOS
# ==============================================================================
enum CorsiState {
	INTRO, 
	CALIBRATING, # Nuevo estado para la medición inicial
	DELAY_BEFORE_SEQUENCE, 
	SHOW_SEQUENCE, 
	USER_INPUT, 
	VERIFY, 
	RESULTS
}
var current_state: CorsiState = CorsiState.INTRO

# ==============================================================================
# TEMPORIZACIÓN
# ==============================================================================
var last_flash_time: float = 0.0
var show_index: int = 0
var show_flash_on: bool = true
var delay_start_time: float = 0.0
var feedback_end_time: float = 0.0
var feedback_color: Color = Color.WHITE

var _game_active: bool = false
var _neuro_track_timer: float = 0.0
const NEURO_TRACK_INTERVAL: float = 0.5

var _energy_saved: bool = false
var _game_completed: bool = false

# ==============================================================================
# FUNCIONES DEL CICLO DE VIDA
# ==============================================================================

func _ready() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	_apply_color_theme()
	_initialize_game()
	
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").register_corsi_game(self)
	
	print("Corsi: Module initialized") # Debug limpio

func _process(delta: float) -> void:
	var now: float = Time.get_ticks_msec() / 1000.0
	
	if _game_active:
		_handle_game_states(now)
		_update_visual_effects(now)
		_update_neuro_feedback()
		_update_ui(now) # Esta función ya está incluida abajo
		_track_neuro_performance(delta)

func _input(event: InputEvent) -> void:
	if event.is_action_pressed("ui_cancel"):
		return_to_world()

# ==============================================================================
# SISTEMA DE CALIBRACIÓN (NUEVO)
# ==============================================================================

func start_calibration_phase():
	print("Corsi: Starting calibration phase")
	_update_game_state(CorsiState.CALIBRATING)
	
	# Ocultamos UI que estorba para que el mensaje sea claro
	progress_bar.hide()
	neuro_bar.hide()
	label_score.hide()
	label_level.hide()
	
	feedback_label.show()
	feedback_label.text = "CALIBRANDO ESCÁNER CEREBRAL...\n(Midiendo carga de memoria de trabajo... 5s)"
	feedback_label.add_theme_color_override("font_color", Color(1.0, 1.0, 0.0))
	
	calibration_buffer.clear()
	is_calibrating = true
	
	await get_tree().create_timer(5.0).timeout
	finish_calibration()

func finish_calibration():
	is_calibrating = false
	
	if calibration_buffer.size() < 10:
		print("Corsi: Warning - Low calibration data. Using defaults.")
		baseline_mean = 0.75
		baseline_std = 0.05
	else:
		# Cálculo de Media
		var sum = 0.0
		for val in calibration_buffer:
			sum += val
		baseline_mean = sum / calibration_buffer.size()
		
		# Cálculo de Desviación Estándar
		var sum_sq_diff = 0.0
		for val in calibration_buffer:
			sum_sq_diff += pow(val - baseline_mean, 2)
		baseline_std = sqrt(sum_sq_diff / calibration_buffer.size())
		baseline_std = max(baseline_std, 0.02) # Evitar 0 absoluto

	# Cálculo de Meta Dinámica
	current_threshold = baseline_mean + (baseline_std * DIFFICULTY_FACTOR)
	current_threshold = clamp(current_threshold, 0.30, 0.92)
	
	print("Corsi: Calibration result - Mean: %.2f, Std: %.2f, Target: %.2f" % [baseline_mean, baseline_std, current_threshold])
	
	feedback_label.text = "¡Escáner Calibrado!"
	feedback_label.add_theme_color_override("font_color", Color(0.0, 1.0, 0.0))
	
	# Restauramos la UI
	progress_bar.show()
	neuro_bar.show()
	label_score.show()
	label_level.show()
	
	# Pausa breve y arranca
	await get_tree().create_timer(1.0).timeout
	feedback_label.text = "" 
	generate_sequence()

# ==============================================================================
# SISTEMA DE ENERGÍA Y NEUROFEEDBACK ADAPTATIVO
# ==============================================================================

func calculate_neuro_bonus_adaptive(current_neuro: float) -> float:
	# Umbral dinámico calculado en calibración
	var target = current_threshold
	var floor_val = baseline_mean - baseline_std
	
	if current_neuro < floor_val:
		return 0.0
		
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
	
	# Usamos el umbral dinámico para determinar "Perfecto"
	if brain_ratio >= current_threshold:
		total_energy += 5
		perfect_neuro_count += 1
	
	return int(max(total_energy, 1))

func save_energy_to_global() -> void:
	if has_node("/root/PlayerState"):
		var player_state = get_node("/root/PlayerState")
		player_state.add_corsi_energy(energy_contribution)
		print("Corsi: Energy saved to global state: ", energy_contribution)
	else:
		print("Corsi: Error - PlayerState not found")

# ==============================================================================
# INICIALIZACIÓN VISUAL
# ==============================================================================

func _initialize_game() -> void:
	_reset_ui()
	_create_blocks()
	_reset_performance_tracking()

func _reset_performance_tracking() -> void:
	energy_contribution = 0
	perfect_neuro_count = 0
	total_neuro_sum = 0.0
	neuro_samples = 0
	correct_sequences = 0
	total_sequences = 0
	sequence_accuracy = 0.0
	level = 1
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
	button.add_theme_color_override("font_color", text_color)

func _reset_ui() -> void:
	label_score.hide()
	label_level.hide()
	progress_bar.hide()
	feedback_label.hide()
	neuro_bar.hide()
	
	button.show()
	menu_label.show()
	menu.show()
	
	# TEXTO MODIFICADO: Contexto + Instrucciones sencillas
	menu_label.text = "🔋 MÓDULO CORSI: REACTOR AUXILIAR\n\n" + \
					  "El núcleo es inestable. Tu memoria de trabajo puede estabilizar el flujo.\n\n" + \
					  "CÓMO JUGAR:\n" + \
					  "1. Observa el orden en que se iluminan los bloques.\n" + \
					  "2. Repite la secuencia exacta haciendo clic.\n" + \
					  "3. Mantén tu mente enfocada para potenciar la energía.\n\n" + \
					  "El sistema iniciará una breve calibración de tu señal cerebral."

func _create_blocks() -> void:
	for child in grid.get_children():
		child.queue_free()
	
	blocks.clear()
	glow_timers.clear()
	
	grid.columns = GRID_SIZE
	grid.add_theme_constant_override("h_separation", 10)
	grid.add_theme_constant_override("v_separation", 10)
	
	var tile_scene: PackedScene = preload("res://Corsi/scenes/CorsiTile.tscn")
	
	for i in range(GRID_SIZE * GRID_SIZE):
		var block: Node = tile_scene.instantiate()
		if block.is_connected("clicked_tile", Callable(self, "_on_block_clicked")):
			block.disconnect("clicked_tile", Callable(self, "_on_block_clicked"))
		block.connect("clicked_tile", Callable(self, "_on_block_clicked"))
		
		block.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		block.size_flags_vertical = Control.SIZE_EXPAND_FILL
		block.custom_minimum_size = Vector2(80, 80)
		
		grid.add_child(block)
		blocks.append(block)
		glow_timers.append(0.0)
		
		block.setup(i % GRID_SIZE, i / GRID_SIZE, i)

# ==============================================================================
# MANEJO DE ESTADOS
# ==============================================================================

func _handle_game_states(now: float) -> void:
	match current_state:
		CorsiState.CALIBRATING:
			# Esperar a que pase el timer en start_calibration_phase
			pass
		CorsiState.DELAY_BEFORE_SEQUENCE:
			_handle_delay_before_sequence(now)
		CorsiState.SHOW_SEQUENCE:
			_handle_show_sequence(now)
		CorsiState.VERIFY:
			_handle_verify_state(now)
		CorsiState.USER_INPUT:
			_handle_user_input_state()

func _handle_delay_before_sequence(now: float) -> void:
	if now - delay_start_time >= START_DELAY:
		last_flash_time = now
		current_state = CorsiState.SHOW_SEQUENCE

func _handle_show_sequence(now: float) -> void:
	var elapsed: float = now - last_flash_time
	if show_flash_on and elapsed >= SHOW_TIME:
		show_flash_on = false
		last_flash_time = now
	elif not show_flash_on and elapsed >= GAP_TIME:
		show_index += 1
		show_flash_on = true
		last_flash_time = now
		if show_index >= sequence.size():
			_transition_to_user_input()

func _transition_to_user_input() -> void:
	_update_game_state(CorsiState.USER_INPUT)
	user_sequence.clear()
	progress_bar.value = 0.0
	_reset_all_blocks()

func _handle_verify_state(now: float) -> void:
	if feedback_end_time != 0 and now >= feedback_end_time:
		if level > MAX_LEVEL: 
			current_state = CorsiState.RESULTS
			_update_results_ui()
		else:
			generate_sequence()
		feedback_end_time = 0

func _handle_user_input_state() -> void:
	var target_value: float = float(user_sequence.size()) / sequence.size() if sequence.size() > 0 else 0.0
	progress_bar.value = lerp(progress_bar.value, target_value, 0.15)

# ==============================================================================
# LÓGICA DEL JUEGO
# ==============================================================================

func generate_sequence() -> void:
	var length: int = 2 + (level - 1)
	sequence.clear()
	user_sequence.clear()
	
	var indices: Array = range(blocks.size())
	indices.shuffle()
	
	for i in range(length):
		sequence.append(indices[i])
	
	show_index = 0
	show_flash_on = true
	delay_start_time = Time.get_ticks_msec() / 1000.0
	_update_game_state(CorsiState.DELAY_BEFORE_SEQUENCE)
	_reset_all_blocks()

func _on_block_clicked(tile: Node) -> void:
	if current_state == CorsiState.USER_INPUT:
		var block_index: int = tile.block_index
		var now: float = Time.get_ticks_msec() / 1000.0
		glow_timers[block_index] = now + GLOW_DURATION
		
		if user_sequence.size() < sequence.size():
			user_sequence.append(block_index)
			progress_bar.value = float(user_sequence.size()) / sequence.size()
			
		if user_sequence.size() >= sequence.size():
			current_state = CorsiState.VERIFY

func verify_sequence() -> void:
	var now: float = Time.get_ticks_msec() / 1000.0
	var is_correct: bool = user_sequence == sequence
	
	total_sequences += 1
	if is_correct: correct_sequences += 1
	
	var energy_earned = calculate_energy_contribution(is_correct)
	
	if is_correct:
		energy_contribution += energy_earned
		level += 1 
		feedback_color = Color(0.0, 0.8, 0.6)
		var neuro_bonus = calculate_neuro_bonus_adaptive(brain_ratio)
		if neuro_bonus > 0:
			feedback_label.text = "✅ Estabilizado! +%d energía | Eficiencia: +%.0f%%" % [energy_earned, neuro_bonus * 100]
		else:
			feedback_label.text = "✅ Estabilizado! +%d energía" % energy_earned
	else:
		energy_contribution += energy_earned
		level += 1 
		feedback_color = Color(0.9, 0.3, 0.3)
		feedback_label.text = "❌ Inestabilidad %d energía" % energy_earned
	
	feedback_end_time = now + 1.2
	feedback_label.add_theme_color_override("font_color", feedback_color)
	
	_update_energy_display()
	
	# AUDIO FIX: Aquí sí queremos que suene, así que pasamos true
	_update_level_display(true) 
	
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").send_module_event("corsi", "sequence_verified", is_correct)

# ==============================================================================
# UI HELPERS (FUNCIONES DE INTERFAZ)
# ==============================================================================

# Esta es la función que faltaba en el paso anterior y causaba el bug
func _update_ui(now: float) -> void:
	match current_state:
		CorsiState.SHOW_SEQUENCE:
			_update_show_sequence_ui()
		CorsiState.USER_INPUT:
			_update_user_input_ui()
		CorsiState.VERIFY:
			_update_verify_ui()
		# Calibración, Delay y Results manejan su propia UI por eventos

func _update_show_sequence_ui() -> void:
	if show_index < sequence.size() and show_flash_on:
		var current_block: int = sequence[show_index]
		for i in blocks.size():
			blocks[i].set_highlighted(i == current_block)
	else:
		for block in blocks:
			block.set_highlighted(false)

func _update_user_input_ui() -> void:
	for i in blocks.size():
		blocks[i].set_selected(i in user_sequence)
	feedback_label.text = "🎯 Replica el patrón: %d/%d" % [user_sequence.size(), sequence.size()]

func _update_verify_ui() -> void:
	if feedback_end_time == 0:
		verify_sequence()

func _update_energy_display() -> void:
	label_score.text = "%d energía" % energy_contribution

# AUDIO FIX: Parámetro añadido para controlar si suena el audio
func _update_level_display(play_sound: bool = false) -> void:
	if level > MAX_LEVEL:
		label_level.text = "Nivel %d/%d" % [MAX_LEVEL, MAX_LEVEL]
		return

	label_level.pivot_offset = label_level.size / 2
	var tween = create_tween()
	tween.tween_interval(0.8)
	tween.tween_property(label_level, "scale", Vector2(1.5, 1.5), 0.2).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
	tween.parallel().tween_property(label_level, "modulate", Color(1.0, 0.8, 0.2), 0.2) 
	tween.tween_callback(func():
		label_level.text = "Nivel %d/%d" % [level, MAX_LEVEL]
		label_level.pivot_offset = label_level.size / 2 
		# Solo reproducir sonido si play_sound es true
		if play_sound and has_node("LevelUpSound"): 
			$LevelUpSound.play() 
	)
	tween.tween_property(label_level, "scale", Vector2(1.0, 1.0), 0.8).set_trans(Tween.TRANS_ELASTIC).set_ease(Tween.EASE_OUT)
	var original_color = Color(0.7, 0.9, 1.0)
	tween.parallel().tween_property(label_level, "modulate", original_color, 0.5)

func _update_results_ui() -> void:
	label_score.hide()
	label_level.hide()
	progress_bar.hide()
	feedback_label.hide()
	neuro_bar.hide()
	
	button.hide()
	menu_label.show()
	menu.show()
	
	if not _energy_saved:
		save_energy_to_global()
		_energy_saved = true
		_game_completed = true
	
	var performance = get_detailed_performance()
	var assessment = get_battery_assessment()
	
	var resultado_texto: String = "🚀 INFORME FINAL - MÓDULO CORSI 🚀\n\n"
	resultado_texto += "⚡ ENERGÍA CONTRIBUIDA: %d/%d\n" % [energy_contribution, MAX_ENERGY]
	resultado_texto += "📊 Estado: %s\n\n" % assessment.status
	resultado_texto += "🎯 ANÁLISIS DE RENDIMIENTO:\n"
	resultado_texto += "• 🎮 Precisión secuencias: %.1f%% (%d/%d)\n" % [performance.sequence_accuracy * 100, performance.correct_sequences, performance.total_sequences]
	resultado_texto += "• 🧠 Memoria Trabajo (Theta/Gamma): %.1f%%\n" % performance.neuro_efficiency
	resultado_texto += "• 🏆 Rendimiento general: %.1f%%\n\n" % (performance.overall_performance * 100)
	resultado_texto += "💡 IMPLICACIONES PARA LA NAVE:\n" + assessment.implication + "\n"
	resultado_texto += "\n🎮 ESC para volver al puente"
	
	menu_label.add_theme_font_size_override("font_size", 20)
	menu_label.text = resultado_texto
	
	_update_game_state(CorsiState.RESULTS)
	
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").send_module_event("corsi", "game_completed", true)

func get_detailed_performance() -> Dictionary:
	var avg_neuro = total_neuro_sum / neuro_samples if neuro_samples > 0 else 0.0
	sequence_accuracy = float(correct_sequences) / total_sequences if total_sequences > 0 else 0.0
	
	return {
		"energy_contribution": energy_contribution,
		"average_neurofeedback": avg_neuro,
		"perfect_moments": perfect_neuro_count,
		"total_samples": neuro_samples,
		"sequence_accuracy": sequence_accuracy,
		"correct_sequences": correct_sequences,
		"total_sequences": total_sequences,
		"neuro_efficiency": avg_neuro * 100,
		"overall_performance": (energy_contribution / 100.0) * 0.6 + (sequence_accuracy) * 0.4
	}

func get_battery_assessment() -> Dictionary:
	var energy = energy_contribution
	var status = ""
	var implication = ""
	var color = Color.WHITE
	
	if energy <= CRITICAL_ENERGY:
		status = "CRÍTICO"
		implication = "Energía insuficiente. Fallos catastróficos posibles."
		color = Color(1.0, 0.2, 0.2)
	elif energy <= LOW_ENERGY:
		status = "BAJO"
		implication = "Energía mínima. Capacidades reducidas."
		color = Color(1.0, 0.6, 0.2)
	elif energy <= ADEQUATE_ENERGY:
		status = "ADECUADO"
		implication = "Energía suficiente para operaciones básicas."
		color = Color(1.0, 0.8, 0.2)
	elif energy <= GOOD_ENERGY:
		status = "BUENO"
		implication = "Buena contribución. Operación eficiente."
		color = Color(0.6, 0.8, 0.2)
	else:
		status = "EXCELENTE"
		implication = "Contribución óptima. Máximo rendimiento."
		color = Color(0.2, 0.8, 0.2)
	
	return {"status": status, "implication": implication, "color": color, "energy_value": energy}

func _reset_all_blocks() -> void:
	for block in blocks:
		block.reset_states()
		block.set_highlighted(false)
		block.set_selected(false)
		block.set_glowing(false)

func _track_neuro_performance(delta: float) -> void:
	if is_calibrating: return
	_neuro_track_timer += delta
	if _neuro_track_timer >= NEURO_TRACK_INTERVAL:
		total_neuro_sum += brain_ratio
		neuro_samples += 1
		_neuro_track_timer = 0.0

func _update_visual_effects(now: float) -> void:
	for i in range(glow_timers.size()):
		blocks[i].set_glowing(glow_timers[i] > now)

func _update_neuro_feedback() -> void:
	neuro_bar.value = brain_ratio

# ==============================================================================
# CONTROL Y COMUNICACIÓN
# ==============================================================================

func start_game() -> void:
	print("Corsi: Game start requested")
	
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").set_game_state("in_corsi_minigame") 
	
	_reset_performance_tracking()
	_game_active = true
	_energy_saved = false
	_game_completed = false
	
	# Mostrar elementos (menos las barras durante calibración)
	# IMPORTANTE: NO mostramos progress_bar ni neuro_bar aquí, se mostrarán tras calibrar
	label_score.show()
	label_level.show()
	
	_update_energy_display()
	
	# AUDIO FIX: false para que NO suene al iniciar
	_update_level_display(false) 
	progress_bar.value = 0
	
	button.hide()
	menu_label.hide()
	menu.hide()
	
	start_calibration_phase()

func _on_button_pressed() -> void:
	start_game()

func return_to_world() -> void:
	if has_node("/root/NeuroFeedbackUDP"):
		get_node("/root/NeuroFeedbackUDP").set_game_state("exploring")
		
	print("Corsi: Returning to world")
	
	if _game_completed and not _energy_saved:
		save_energy_to_global()
		_energy_saved = true
	
	get_tree().change_scene_to_file("res://main.tscn")
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func set_brain_ratio(ratio: float) -> void:
	if ratio < 0.01 or ratio > 1.0: return
	
	brain_ratio = ratio
	
	if is_calibrating:
		calibration_buffer.append(ratio)

func _update_game_state(new_state: CorsiState):
	if current_state != new_state:
		current_state = new_state
		var state_str = ""
		match new_state:
			CorsiState.INTRO: state_str = "INTRO"
			CorsiState.CALIBRATING: state_str = "CALIBRATING"
			CorsiState.DELAY_BEFORE_SEQUENCE: state_str = "DELAY"
			CorsiState.SHOW_SEQUENCE: state_str = "SHOW"
			CorsiState.USER_INPUT: state_str = "INPUT"
			CorsiState.VERIFY: state_str = "VERIFY"
			CorsiState.RESULTS: state_str = "RESULTS"
	
		if has_node("/root/NeuroFeedbackUDP"):
			get_node("/root/NeuroFeedbackUDP").send_minigame_state("corsi", state_str)
