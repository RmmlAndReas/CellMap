"""Drag and drop event handler."""


def handle_drag_enter(main_window, event):
    """Handle drag enter event."""
    selected_tab_idx, _ = main_window.get_cur_tab_index_and_name()
    list_idx = main_window._tab_idx_to_list_idx(selected_tab_idx)
    main_window.list.get_list(list_idx).dragEnterEvent(event)


def handle_drag_move(main_window, event):
    """Handle drag move event."""
    selected_tab_idx, _ = main_window.get_cur_tab_index_and_name()
    list_idx = main_window._tab_idx_to_list_idx(selected_tab_idx)
    main_window.list.get_list(list_idx).dragMoveEvent(event)


def handle_drop(main_window, event):
    """Handle drop event."""
    selected_tab_idx, _ = main_window.get_cur_tab_index_and_name()
    list_idx = main_window._tab_idx_to_list_idx(selected_tab_idx)
    # Ensure the correct list is visible before dropping
    main_window.list.set_list(list_idx)
    main_window.list.get_list(list_idx).dropEvent(event)
