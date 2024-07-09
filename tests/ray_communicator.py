# tests/test_ray_communicator.py

import ray
from communicators.ray_communicator import RayCommunicator

context = ray.init(ignore_reinit_error=True)
print(context.dashboard_url)


def test_ray_communicator():
    # 创建RayCommunicator实例
    names = ['comm1', 'comm2', 'comm3']
    communicator_1 = RayCommunicator.options(name=names[0]).remote(name=names[0])
    communicator_2 = RayCommunicator.options(name=names[1]).remote(name=names[1])
    communicator_3 = RayCommunicator.options(name=names[2]).remote(name=names[2])

    # 测试 neighbors_send 和 receive
    test_message = "Hello, Ray!"
    ray.get(communicator_1.send_to_nbrs.remote(test_message, [communicator_2, communicator_3]))

    # 获取inbox中的消息
    inbox_2 = ray.get(communicator_2.get_inbox.remote())
    inbox_3 = ray.get(communicator_3.get_inbox.remote())

    assert inbox_2 == [("comm1", test_message)], f"Expected [('comm1', '{test_message}')], but got {inbox_2}"
    assert inbox_3 == [("comm1", test_message)], f"Expected [('comm1', '{test_message}')], but got {inbox_3}"

    # 测试 neighbors_receive
    response_1 = ray.get(communicator_1.neighbors_receive.remote([communicator_2, communicator_3]))

    expected_response = {"comm2": "ready", "comm3": "ready"}
    assert response_1 == expected_response, f"Expected {expected_response}, but got {response_1}"

    # 测试 neighbors_exchange
    exchange_message = "Exchange Message"
    exchange_result = ray.get(
        communicator_1.neighbors_exchange.remote(exchange_message, [communicator_2, communicator_3],
                                                 [communicator_2, communicator_3]))

    assert exchange_result == {"comm2": "ready",
                               "comm3": "ready"}, f"Expected {{'comm2': 'ready', 'comm3': 'ready'}}, but got {exchange_result}"

    print("All tests passed!")


if __name__ == "__main__":
    test_ray_communicator()