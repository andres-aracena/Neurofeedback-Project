from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

def init_board(board_id=BoardIds.SYNTHETIC_BOARD.value):
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    print("Placa conectada.")
    return board

def get_eeg_channels(board, N_CH):
    eeg_channels = board.get_eeg_channels(board.get_board_id())
    return eeg_channels[:N_CH]
