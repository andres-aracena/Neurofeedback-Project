# main3d.gd - Escena principal
extends Node3D

func _ready():
	setup_global_illumination()
	
	print("Mundo 3D cargado")
	# Asegurar que el mouse este capturado
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
	
	# Buscar el jugador en la escena
	var player = find_child("Player")  # Ajusta el nombre segun tu escena
	if player:
		print("Jugador encontrado en el mundo 3D")
	else:
		print("No se encontro el jugador en la escena principal")

func setup_global_illumination():
	# Opción 1: Usar SDFGI (más fácil)
	var env = $WorldEnvironment.environment
	if env:
		env.sdfgi_enabled = true
		env.sdfgi_energy = 1.0
   
