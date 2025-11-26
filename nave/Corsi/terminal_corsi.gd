extends Area3D

@onready var hint_label: Label3D = $Label3D
@export var game_scene: PackedScene = preload("res://Corsi/scenes/CorsiGame.tscn")

var player_ref: Node = null
var can_interact: bool = false

func _ready():
	monitoring = true
	hint_label.visible = false
	
	if not body_entered.is_connected(_on_area3d_body_entered):
		body_entered.connect(_on_area3d_body_entered)
	if not body_exited.is_connected(_on_area3d_body_exited):
		body_exited.connect(_on_area3d_body_exited)

func _on_area3d_body_entered(body: Node) -> void:
	if body.is_in_group("player"):
		player_ref = body
		hint_label.visible = true
		can_interact = true

func _on_area3d_body_exited(body: Node) -> void:
	if body == player_ref:
		player_ref = null
		hint_label.visible = false
		can_interact = false

func _process(delta):
	if can_interact and Input.is_action_just_pressed("ui_letter_e"):
		start_game()

func start_game() -> void:
	if game_scene == null:
		push_error("terminal_corsi: game_scene no asignada")
		return
	
	print("Iniciando juego N-back...")
	
	# GUARDAR ESTADO DEL JUGADOR ANTES DE SALIR
	if player_ref:
		PlayerState.save_player_state(player_ref)  # Cambiado a PlayerState
	
	# Liberar el mouse antes de cambiar de escena
	Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	
	# Desactivar input del jugador
	if player_ref and player_ref.has_method("disable_input"):
		player_ref.disable_input()
	
	# Cambiar a la escena del juego N-back
	get_tree().change_scene_to_packed(game_scene)
