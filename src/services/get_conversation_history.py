import redis
import json
import logging

logging.basicConfig(level=logging.INFO)

def get_conversation_window(user_id: str) -> str:
    """Get recent conversation context from Redis"""
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Pattern to get all checkpoint (conversations) keys for a user
        pattern = f"checkpoint:{user_id}:__empty__:*"

        # all checkpoints for a user
        keys = redis_client.keys(pattern)
        if not keys:
            return "No conversation history"
        
        converted_checkpoints = []
        # convert from REDISJSON to python object
        for key in keys:
            raw_value = redis_client.execute_command("JSON.GET", key)
            data = json.loads(raw_value)
            converted_checkpoints.append(data)

        # sort by ascending timestamp
        sorted_checkpoints = sorted(converted_checkpoints, key=lambda x: x.get("checkpoint", {}).get("ts", ""))

        latest_checkpoint = sorted_checkpoints[-1]

        messages = latest_checkpoint.get("checkpoint", {}).get("channel_values", {}).get("messages", [])
        
        # Extract just the content from each message
        message_contents = [msg.get("kwargs", {}).get("content", "") for msg in messages]
        result = latest_checkpoint.get("checkpoint", {}).get("channel_values", {}).get("result", {}).get("smartrouter_result", "")
        return {"user_messages": message_contents, "agent_result": result}

    except Exception as e:
        return f"Error: {e}"

result = get_conversation_window("matt1234")
print(result)