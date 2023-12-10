import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

INFO_FUNCS = {
    "high": lambda x: max(x),
    "low": lambda x: min(x),
    "avg": lambda x: float(sum(x)) / len(x)
}

def split_list(info, chunk_size: int) -> list:
    """
        Function splits an iterable (into) into chunk_sized sublists

        :param info: (iterable) info to split
        :param chunk_size: (int) size of each sublist
        :return: (list) list of sublists each of size chunk_size
    """
    return [info[x: x + chunk_size] for x in range(0, len(info), chunk_size)]

def get_list_info(info) -> dict:
    """
        Function returns data info for provided iterable

        :param info: (iterable) info to process
        :return: (dict) {info type: info values}
    """
    return {func_key: func(info) for func_key, func in INFO_FUNCS.items()}

def get_split_list_info(split_list: list) -> np.ndarray:
    """
        Function returns an numpy ndarray with data for a iterable with sublists to process

        :param split_list: (list of sublists) sublists to grab info for
        :return: (numpy ndarray) all info of provided sublists
    """
    return np.array([list(get_list_info(ind_list).values()) for ind_list in split_list])

def create_animation(info, chunk: int, interval: int, frame_chunk: int = None) -> FuncAnimation:
    """
        This function creates a graph animation of the provided info.

        :param info: (iterable) data to iterate through and display on the graph
        :param chunk: (int) chunk of info to show on one frame
        :param interval: (int) interval in ms to update the animation
        :param frame_chunk: (int, optional) chunk of information to swap on each frame update
                                            if not provided, will update one piece of info at each frame update
        :return: (FuncAnimation) animation showing data using provided parameters
    """
    fig, ax = plt.subplots()
    ax.set_ylim([0, max(info)])

    def update(frame):
        x = list(range(frame, frame + chunk))
        y = info[frame: frame + chunk]
        ax.set_xlim([x[0], x[-1]])
        ax.plot(x, y, color='blue')

    if frame_chunk is None:
        return FuncAnimation(fig, update, frames=len(info) - chunk, repeat=False, interval=interval)
    else:
        return FuncAnimation(fig, update, frames=range(0, len(info) - chunk, frame_chunk), repeat=False,
                             interval=interval)

def create_animation_w_sublists(info, info_scalar: int, chunk: int, interval: int,
                                frame_chunk: int = None) -> FuncAnimation:
    """
        This function creates a graph animation of the provided info.

        :param info: (iterable with columns) data to iterate through and display on the graph
                                             should be same as data returned by get_split_list_info
        :param info_scalar: (int) scalar used to scale up X-axis
        :param chunk: (int) chunk of info to show on one frame
        :param interval: (int) interval in ms to update the animation
        :param frame_chunk: (int, optional) chunk of information to swap on each frame update
                                            if not provided, will update one piece of info at each frame update
        :return: (FuncAnimation) animation showing data using provided parameters
    """
    fig, ax = plt.subplots()
    item_dict = {'High': 'red', 'Low': 'black', 'Average': 'blue'}
    ax.set_ylim([0, np.amax(info)])
    ax.set_title(f'Scores for every 100 episodes')

    def update(frame):
        x = list(range(frame * info_scalar, (frame + chunk) * info_scalar, info_scalar))
        y = info[frame: frame + chunk, :]
        ax.set_xlim([x[0], x[-1]])
        for i, (val_type, color) in enumerate(item_dict.items()):
            ax.plot(x, y[:, i], label=val_type, color=color)

        ax.legend(item_dict.keys(), loc='upper right')

    if frame_chunk is None:
        return FuncAnimation(fig, update, frames=len(info) - chunk, repeat=False, interval=interval)
    else:
        return FuncAnimation(fig, update, frames=range(0, len(info) - chunk, frame_chunk), repeat=False,
                             interval=interval)

def create_plot(info, chunk: int = 100) -> None:
    """
        This function creates a simple plot summarizing data for chunks of info.

        :param info: (iterable) data to split into pieces of chunk size and grab info for.
        :param chunk: (int) info will be split into pieces of chunk size
                            data will be provided for each chunk of info
        :return: None
    """
    fig, ax = plt.subplots()
    split_info = split_list(info, chunk)
    assert len(split_info) * len(split_info[0]) == len(info)
    info_vals = get_split_list_info(split_info)
    max_val = np.amax(info_vals)
    for i, val_type in enumerate(['high', 'low', 'avg']):
        ax.plot([x for x in range(0, len(info), chunk)], info_vals[:, i], label=val_type)
    ax.set_title(f'Stats for every {chunk} episodes')
    # values used for consistent y scaling
    ax.set_ylim([-250.0, 6790.0])
    ax.legend()
    print(max_val, ax.get_ylim())

if __name__ == "__main__":
    with open('Scores/MsPacMan_training_scores.csv', mode='r') as csv_file:
        chunk = 100
        info = list(csv.reader(csv_file))[0]
        info = [float(x) for x in info]
        create_plot(info, chunk)
        plt.show()

        ani = create_animation(info, chunk, 200, 25)
        # ani.save('Gifs/pac_man1.gif', writer='pillow')
        plt.show()

        split_info = split_list(info, chunk)
        split_info_data = get_split_list_info(split_info)
        ani = create_animation_w_sublists(split_info_data, chunk, 10, 200)
        # ani.save('Gifs/pac_man2.gif', writer='pillow')
        plt.show()


    with open('Scores/scores_beforeOptimizing.csv', mode='r') as csv_file:
        chunk = 100
        info = list(csv.reader(csv_file))[0]
        info = [float(x) for x in info]
        create_plot(info, chunk)
        plt.show()