# tests/test_ray_communicator.py

import pytest
import ray
from communicators.ray_communicator import RayCommunicator

@pytest.fixture(scope="module")
def ray_start():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    yield
    # Shutdown Ray
    ray.shutdown()

@pytest.fixture
def setup_agents():
    # Create RayCommunicator instances for testing
    agent1 = RayCommunicator.options(name="1").remote(1, [], [2])
    agent2 = RayCommunicator.options(name="2").remote(2, [1], [])
    return agent1, agent2

def test_get_ids(ray_start, setup_agents):
    agent1, agent2 = setup_agents
    agent1_ids = ray.get(agent1.get_ids.remote())
    agent2_ids = ray.get(agent2.get_ids.remote())
    assert agent1_ids == (1, [], [2])
    assert agent2_ids == (2, [1], [])

def test_set_cached_neighbors(ray_start, setup_agents):
    agent1, agent2 = setup_agents
    ray.wait([agent1.set_cached_neighbors.remote(), agent2.set_cached_neighbors.remote()])
    cached_in_handles_agent1 = ray.get(agent1.get_cached_in_handles.remote())
    cached_out_handles_agent1 = ray.get(agent1.get_cached_out_handles.remote())
    cached_in_handles_agent2 = ray.get(agent2.get_cached_in_handles.remote())
    cached_out_handles_agent2 = ray.get(agent2.get_cached_out_handles.remote())
    assert cached_in_handles_agent1 == {}
    assert cached_out_handles_agent1 == {2: ray.get_actor("2")}
    assert cached_in_handles_agent2 == {1: ray.get_actor("1")}
    assert cached_out_handles_agent2 == {}

def test_recv_from_nbr(ray_start, setup_agents):
    agent1, agent2 = setup_agents
    msg = "Hello from agent 1"
    sender_id, received_msg = ray.get(agent2.recv_from_nbr.remote(1, msg))
    assert sender_id == 1
    assert received_msg == msg
    inbox_agent2 = ray.get(agent2.get_inbox.remote())
    assert inbox_agent2 == {1: msg}

def test_set_outbox(ray_start, setup_agents):
    agent1, agent2 = setup_agents
    msgs = {2: "Message to agent 2"}
    ray.get(agent1.set_outbox.remote(msgs))
    # outbox_agent1 = ray.get(ray.wait(agent1.get_outbox.remote()))  # also work
    outbox_agent1 = ray.get(agent1.get_outbox.remote())
    assert outbox_agent1 == msgs

def test_send_to_nbrs(ray_start, setup_agents):
    agent1, agent2 = setup_agents
    ray.wait([agent1.set_cached_neighbors.remote(), agent2.set_cached_neighbors.remote()])

    msgs = {2: "Broadcast message from agent 1"}
    ray.get(agent1.set_outbox.remote(msgs))
    ray.get(agent1.send_to_nbrs.remote([2]))
    inbox_agent2 = ray.get(agent2.get_inbox.remote())
    assert inbox_agent2 == {1: "Broadcast message from agent 1"}

def test_check_subset():
    valid_list = [1, 2, 3]
    subset = [1, 2]
    invalid_subset = [4]

    RayCommunicator.check_subset(subset, valid_list)  # Should not raise any exception

    with pytest.raises(RuntimeError):
        RayCommunicator.check_subset(invalid_subset, valid_list)  # Should raise exception