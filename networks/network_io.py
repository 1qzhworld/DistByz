# networks/network_io.py

import os
import joblib
from utils.utils import common_networks_dir
from networks.network import Network


class NetworkIO:
    @staticmethod
    def save_common_network(common_network: Network, filename: str) -> None:
        """
        Save the common network to a file.

        Args:
            common_network (Network): The network object to save.
            filename (str): The name of the file to save the network to.

        Raises:
            IOError: If there is an issue saving the file.
        """
        try:
            save_path = os.path.join(common_networks_dir, filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(common_network, save_path)
            print(f"Saved common network to: {save_path}")
        except IOError as e:
            print(f"Error saving network to file: {e}")
            raise

    @staticmethod
    def load_common_network(filename: str) -> Network:
        """
        Load the common network from a file.

        Args:
            filename (str): The name of the file to load the network from.

        Returns:
            Network: The loaded network object.

        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If there is an issue loading the file.
        """
        load_path = os.path.join(common_networks_dir, filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Common network file not found: {load_path}")

        try:
            common_network = joblib.load(load_path)
            return common_network
        except IOError as e:
            print(f"Error loading network from file: {e}")
            raise
