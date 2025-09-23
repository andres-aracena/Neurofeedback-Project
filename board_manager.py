# board_manager.py
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds


def init_board(use_synthetic=False, serial_port="COM4"):
    """
    Inicializa la board:
      - use_synthetic=True → usa señal artificial (Synthetic Board).
      - use_synthetic=False → usa Cyton conectado por el puerto serial.
    """
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()

    if use_synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD.value
        print("[INFO] Usando señal SINTÉTICA.")
    else:
        board_id = BoardIds.CYTON_BOARD.value
        params.serial_port = serial_port
        print(f"[INFO] Usando Cyton en puerto {serial_port}.")

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    print("[INFO] Board conectada correctamente.")

    return board


def get_eeg_channels(board, N_CH):
    """
    Devuelve la lista de canales EEG ajustada a N_CH.
    Si la board tiene menos canales, repite para completar.
    """
    eeg_channels = board.get_eeg_channels(board.get_board_id())

    if len(eeg_channels) >= N_CH:
        return eeg_channels[:N_CH]
    else:
        times = int(np.ceil(N_CH / len(eeg_channels)))
        eeg_channels = (eeg_channels * times)[:N_CH]
        print(f"[WARN] La board tiene menos canales, se repitieron para llegar a {N_CH}.")
        return eeg_channels
