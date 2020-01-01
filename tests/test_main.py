from unittest import TestCase
import actor_critic
import mock

mockEnv = mock.MagicMock()
mockEnv.observation_space.shape = (2, 2)
mockEnv.action_space.shape = (2, 2)

class TestActorCritic(TestCase):
    def test_create_actor_model(self):
        agent = actor_critic.ActorCritic(mockEnv, None)
        res = agent.create_actor_model()
        self.assertEqual(len(res), 2)
