import json
import boto3

def test_bedrock_chat():
    # Initialize the Bedrock runtime client
    bedrock_runtime = boto3.client('bedrock-runtime')
    
    # Create a simple chat message payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": "Hello world, tell me a joke?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        # Call Bedrock
        response = bedrock_runtime.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',  # or: anthropic.claude-instant-v1
            body=json.dumps(payload)
        )
        
        # Parse and print the response
        response_body = json.loads(response.get('body').read())
        print("Response from Claude:")
        print(response_body['content'][0]['text'])
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_bedrock_chat()