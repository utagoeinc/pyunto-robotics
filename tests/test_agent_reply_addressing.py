"""The robot's own short replies are addressed to whoever instructed it.

Posted to the room, "I did not understand that" reads as an entry for everyone, and a text
agent in the same diary would answer it.
"""
from pyunto_agent.client import IncomingMessage

from pyunto_robotics.agent import RobotAgent


class FakeClient:
    def __init__(self):
        self.sent = []

    def send(self, chat_space_id, text, **kw):
        self.sent.append((chat_space_id, text, kw))


class Skills:
    def run(self, action, argument=None, where=None, expect=None):  # pragma: no cover
        raise AssertionError("not reached")


def test_short_reply_notifies_the_person_who_wrote():
    client = FakeClient()
    agent = RobotAgent(robot=None, grounder=None, client=client, skills=Skills())
    m = IncomingMessage(uuid="m1", text="make me a coffee", thread_id="t1",
                        chat_space_id="s1", sender_uuid="alice", sender_name="Alice")
    agent._reply(m, "I did not understand that.")
    (_, _, kw), = client.sent
    assert kw["notify_users"] == ["alice"]
    assert kw["thread_id"] == "t1"
