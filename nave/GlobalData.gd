# GlobalData.gd
class_name GlobalData

static var player_position: Vector3 = Vector3.ZERO
static var player_rotation: Vector3 = Vector3.ZERO
static var player_camera_mode: bool = true
static var player_camera_position: Vector3 = Vector3(0, 0.5, 0)
static var has_saved_data: bool = false

static func save_player_data(position: Vector3, rotation: Vector3, camera_mode: bool, camera_position: Vector3) -> void:
	player_position = position
	player_rotation = rotation
	player_camera_mode = camera_mode
	player_camera_position = camera_position
	has_saved_data = true
	print("Estado del jugador guardado:")
	print("   - Posicion: ", player_position)
	print("   - Rotacion: ", player_rotation)
	print("   - Modo camara: ", "Primera persona" if player_camera_mode else "Tercera persona")

static func load_player_data() -> Dictionary:
	if has_saved_data:
		return {
			"position": player_position,
			"rotation": player_rotation,
			"camera_mode": player_camera_mode,
			"camera_position": player_camera_position,
			"has_data": true
		}
	else:
		return {"has_data": false}

static func clear_data() -> void:
	has_saved_data = false
