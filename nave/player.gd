extends CharacterBody3D

@export var speed: float = 5.0
@export var jump_velocity: float = 8.0
@export var gravity: float = 20.0
@export var descend_velocity: float = -8.0
@export var mouse_sensitivity: float = 0.1

# Variables para el control de cámara
@export var first_person_mode: bool = true
@export var camera_switch_key: String = "c"

# Configuración de FOV
@export var first_person_fov: float = 85.0
@export var third_person_fov: float = 70.0

# Referencias a nodos
@onready var pivot = $Pivot
@onready var camera = $Pivot/Camera3D
@onready var mesh_instance = $MeshInstance3D
@onready var game_ui: CanvasLayer = $GameUI

# Variables de rotación
var rotation_y := 0.0
var rotation_x := 0.0
var input_enabled: bool = true

# Neurofeedback simulation - OPTIMIZADO
var neuro_timer: float = 0.0
var neuro_update_interval: float = 0.1

# Configuración de cámaras
@export var first_person_position = Vector3(0, 0.5, 0) : 
	set(value):
		first_person_position = value
		if first_person_mode and camera:
			camera.position = first_person_position

@export var third_person_spring_length = 4.0

# Control del mouse
var mouse_captured: bool = true

# Cache para optimización - NUEVO
var last_context_message: String = ""
var context_message_cooldown: float = 0.0

func _ready():
	print("Player inicializando...")
	
	# Verificar que los nodos existan
	if pivot == null:
		push_error("Nodo Pivot no encontrado!")
	if camera == null:
		push_error("Nodo Camera3D no encontrado!")
	
	# CARGAR ESTADO DEL JUGADOR SI EXISTE
	PlayerState.load_player_state(self)
	
	# Asegurar que el mouse este capturado al inicio
	capture_mouse(true)
	add_to_group("player")
	
	# Cargar UI si no existe
	if not has_node("GameUI"):
		var ui_scene = preload("res://GameUI.tscn")
		if ui_scene:
			game_ui = ui_scene.instantiate()
			add_child(game_ui)
			print("UI cargada correctamente")
		else:
			push_error("No se pudo cargar la escena GameUI")
	
	# Configurar camara segun el modo inicial
	update_camera_mode()
	
	print("Player inicializado correctamente")

func load_saved_state() -> void:
	"""Carga el estado guardado del jugador desde GlobalData"""
	var saved_data = GlobalData.load_player_data()
	if saved_data["has_data"]:
		print("Aplicando datos guardados del jugador...")
		
		global_position = saved_data["position"]
		rotation = saved_data["rotation"]
		first_person_mode = saved_data["camera_mode"]
		first_person_position = saved_data["camera_position"]
		
		update_camera_mode()
		
		print("Estado cargado - Posicion: ", global_position)
		
		# Limpiar datos después de usarlos
		GlobalData.clear_data()
	else:
		print("No hay datos guardados del jugador")

func _process(delta):
	# Simular datos de neurofeedback
	neuro_timer += delta
	if neuro_timer >= neuro_update_interval:
		neuro_timer = 0.0
		simulate_neurofeedback()
	
	# Actualizar mensajes contextuales con control de frecuencia - OPTIMIZADO
	context_message_cooldown += delta
	if context_message_cooldown >= 0.5:  # Cada 500ms
		update_contextual_messages()
		context_message_cooldown = 0.0
	
	# Asegurar que la camara mantenga la posicion correcta
	if first_person_mode and camera:
		camera.position = first_person_position

func capture_mouse(capture: bool):
	if capture:
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
		mouse_captured = true
	else:
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		mouse_captured = false

func simulate_neurofeedback():
	"""Simula neurofeedback para testing - OPTIMIZADO"""
	if game_ui:
		var random_change = randf_range(-0.05, 0.05)
		var new_ratio = clamp(game_ui.brain_ratio + random_change, 0.1, 0.95)
		game_ui.set_brain_ratio(new_ratio)

func update_contextual_messages():
	"""Actualiza mensajes contextuales optimizado - NUEVA VERSIÓN"""
	if game_ui:
		var new_message = ""
		
		if is_on_floor():
			if Input.is_action_pressed("ui_focus_next"):
				new_message = "Descendiendo... Manten SHIFT para bajar mas rapido"
			elif velocity.length() > 2.0:
				new_message = "Movimiento activo - Usa el MOUSE para rotar la camara"
			else:
				new_message = "🎮 Mov: W A S D | ✨ Saltar: Espacio | 📷 Cámara: C "
		else:
			new_message = "En el aire - ESPACIO para saltar nuevamente"
		
		# Solo actualizar si el mensaje cambió - OPTIMIZACIÓN
		if new_message != last_context_message:
			game_ui.show_instructions(new_message)
			last_context_message = new_message

func _input(event):
	# Manejo del ESC para liberar/capturar mouse
	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		if mouse_captured:
			capture_mouse(false)
			if game_ui:
				game_ui.show_message("Mouse Liberado - Presiona ESC para continuar", 3.0)
		else:
			capture_mouse(true)
			if game_ui:
				game_ui.show_message("Mouse Capturado", 1.0)
		get_viewport().set_input_as_handled()
	
	# Cambiar modo de camara
	if event is InputEventKey and event.pressed and event.keycode == KEY_C and mouse_captured:
		toggle_camera_mode()
		get_viewport().set_input_as_handled()
	
	# Movimiento del mouse
	if event is InputEventMouseMotion and mouse_captured and input_enabled:
		var mouse_movement = event.relative * mouse_sensitivity
		
		# Rotacion horizontal (Y) - siempre sin limites
		rotation_y -= mouse_movement.x
		
		# Rotacion vertical (X) - con limites segun el modo
		if first_person_mode:
			rotation_x = clamp(rotation_x - mouse_movement.y, -90, 90)
		else:
			rotation_x = clamp(rotation_x - mouse_movement.y, -45, 45)
		
		# Aplicar rotacion al pivot
		if pivot:
			pivot.rotation_degrees = Vector3(rotation_x, rotation_y, 0)
		
		get_viewport().set_input_as_handled()

func _physics_process(delta):
	if not input_enabled:
		if not is_on_floor():
			velocity.y -= gravity * delta
		else:
			velocity.y = 0
		move_and_slide()
		return
		
	# Obtener direcciones de la camara
	var cam_forward = -pivot.global_transform.basis.z
	var cam_right = pivot.global_transform.basis.x
	
	# Input de movimiento
	var move_input = Vector2.ZERO
	if Input.is_action_pressed("ui_right") or Input.is_key_pressed(KEY_D):
		move_input.x += 1
	if Input.is_action_pressed("ui_left") or Input.is_key_pressed(KEY_A):
		move_input.x -= 1
	if Input.is_action_pressed("ui_down") or Input.is_key_pressed(KEY_S):
		move_input.y -= 1
	if Input.is_action_pressed("ui_up") or Input.is_key_pressed(KEY_W):
		move_input.y += 1

	move_input = move_input.normalized()

	# Calcular direccion de movimiento
	var move_dir = (cam_forward * move_input.y + cam_right * move_input.x).normalized()
	move_dir.y = 0

	# Aplicar velocidad
	velocity.x = move_dir.x * speed
	velocity.z = move_dir.z * speed

	# Gravedad
	if not is_on_floor():
		velocity.y -= gravity * delta
	else:
		velocity.y = 0

	# Salto
	if (Input.is_action_just_pressed("ui_accept") or Input.is_key_pressed(KEY_SPACE)) and is_on_floor():
		velocity.y = jump_velocity

	# Descender
	if Input.is_action_pressed("ui_focus_next") or Input.is_key_pressed(KEY_SHIFT):
		velocity.y = descend_velocity

	move_and_slide()

func toggle_camera_mode():
	first_person_mode = !first_person_mode
	update_camera_mode()
	
	if game_ui:
		if first_person_mode:
			game_ui.show_message("VISTA EN PRIMERA PERSONA ACTIVADA", 2.0)
		else:
			game_ui.show_message("VISTA EN TERCERA PERSONA ACTIVADA", 2.0)

func update_camera_mode():
	if first_person_mode:
		# Modo primera persona
		if camera:
			camera.position = first_person_position
			camera.fov = first_person_fov
		if mesh_instance:
			mesh_instance.visible = false
		if pivot is SpringArm3D:
			pivot.spring_length = 0.1
	else:
		# Modo tercera persona
		if camera:
			camera.position = Vector3(0, 0, 0)
			camera.fov = third_person_fov
		if pivot is SpringArm3D:
			pivot.spring_length = third_person_spring_length
		else:
			if camera:
				camera.position = Vector3(0, 2, 4)
		if mesh_instance:
			mesh_instance.visible = true
	
	if camera:
		camera.near = 0.05

func disable_input():
	input_enabled = false
	velocity = Vector3.ZERO

func enable_input():
	input_enabled = true

# Funcion para cambiar la altura de la camara manualmente
func set_camera_height(height: float):
	first_person_position.y = height

# Funciones auxiliares para GlobalData
func set_first_person_mode(mode: bool) -> void:
	first_person_mode = mode
	update_camera_mode()

func get_first_person_mode() -> bool:
	return first_person_mode
