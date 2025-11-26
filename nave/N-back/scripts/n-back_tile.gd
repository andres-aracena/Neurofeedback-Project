extends Button

signal clicked_tile(tile)

var row: int = -1
var col: int = -1
var block_index: int = -1

# Estados para el juego Corsi
var is_highlighted: bool = false
var is_glowing: bool = false
var is_selected: bool = false

@onready var color_rect: ColorRect = $ColorRect

func _ready():
	print("Tile ", block_index, " created and ready")
	color_rect.hide()
	# Asegurar que el botón sea interactivo
	disabled = false
	focus_mode = FOCUS_ALL
	mouse_filter = MOUSE_FILTER_PASS

func setup(r: int, c: int, index: int):
	row = r
	col = c
	block_index = index
	print("Tile setup: index=", index, " position=(", r, ",", c, ")")
	update_visual()

func update_visual():
	if is_glowing:
		modulate = Color(0, 1, 0.8)  # Cian brillante para glow
	elif is_selected:
		modulate = Color(0.4, 1, 0.8)  # Verde cian para seleccionado
	elif is_highlighted:
		modulate = Color(0, 0.9, 1)  # Cian para resaltado
	else:
		modulate = Color(0.12, 0.16, 0.22)  # Gris azulado oscuro para normal

func _on_pressed():
	print("Tile ", block_index, " pressed - emitting signal")
	emit_signal("clicked_tile", self)

func highlight(on: bool):
	color_rect.visible = on

# Nuevos métodos específicos para el juego Corsi
func set_highlighted(highlight: bool):
	is_highlighted = highlight
	update_visual()

func set_glowing(glow: bool):
	is_glowing = glow
	update_visual()

func set_selected(selected: bool):
	is_selected = selected
	update_visual()

# Función para resetear todos los estados
func reset_states():
	is_highlighted = false
	is_glowing = false
	is_selected = false
	color_rect.hide()
	update_visual()
