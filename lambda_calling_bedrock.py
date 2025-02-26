import json
import boto3

def lambda_handler(event, context):
    # Initialize Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )
    
    # Configure model parameters
    model_id = 'cohere.command-text-v14'
    prompt = "who is the best cricket player"
    
    try:
        # Invoke Cohere model
        response = bedrock.invoke_model(
            body=json.dumps({
                "prompt": prompt,
                "max_tokens": 200,
                "temperature": 0.5
            }),
            modelId=model_id,
            accept='application/json',
            contentType='application/json'
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        answer = response_body['generations'][0]['text']
        
        return {
            'statusCode': 200,
            'body': answer.strip()
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': f"Error: {str(e)}"
        }
