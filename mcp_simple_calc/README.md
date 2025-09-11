# Simple MCP Calculator (FastMCP on AWS Lambda + Docker)

This project runs a [FastMCP](https://pypi.org/project/fastmcp/) server that exposes simple calculator tools (`add`, `subtract`, `multiply`, `divide`) and a `greeting` resource.
It is packaged as a Docker container that works with **AWS Lambda Web Adapter** and can also be tested **locally**.

---

## Project structure

```
.
├── Dockerfile
├── requirements.txt
└── simple_mcp_calculator.py
```

---

## Running locally with Docker

1. **Build the image**:

   ```bash
   docker build -t simple-mcp .
   ```

2. **Run the container** (maps container port 8000 → local port 8000):

   ```bash
   docker run -p 8000:8000 simple-mcp
   ```

4. **Use with MCP client (e.g., VS Code)**
   Update your client config:

   ```json
    {
        "mcpServers": {
            "local_simple_mcp_calculator": {
                "transport": "http",
                "url": "http://127.0.0.1:8000/mcp"
            }
        }
    }
   ```

---

## Deploying to AWS Lambda

This project is containerized for **Lambda Web Adapter**, which makes it behave like an HTTP server.

1. **Authenticate with ECR**:
   ```bash
   aws ecr get-login-password --region <region>      | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
   ```

2. **Create an ECR repository**:
   ```bash
   aws ecr create-repository --repository-name simple-mcp
   ```

3. **Build and push the image**:
   ```bash
   docker build -t simple-mcp .
   docker tag simple-mcp:latest <account_id>.dkr.ecr.<region>.amazonaws.com/simple-mcp:latest
   docker push <account_id>.dkr.ecr.<region>.amazonaws.com/simple-mcp:latest
   ```

4. **Create the Lambda function**:
   - Runtime: **Provide your own container**
   - Image: `<account_id>.dkr.ecr.<region>.amazonaws.com/simple-mcp:latest`

   The Lambda Web Adapter will automatically expose your FastMCP app.

5. **Expose via API Gateway**:
   - Create an HTTP API in API Gateway
   - Integrate it with your Lambda function
   - Deploy the API
   - Your MCP server will now be reachable at:
     ```
     https://<api-id>.execute-api.<region>.amazonaws.com/mcp
     ```

---

## MCP Client Config for Lambda

After deployment, update your MCP client config:

```json
{
  "mcpServers": {
    "calc": {
      "transport": "http",
      "url": "https://<api-id>.execute-api.<region>.amazonaws.com/mcp"
    }
  }
}
```

---

## Notes

- `app = mcp.http_app()` is required in `simple_mcp_calculator.py` for Lambda Web Adapter.
- The `if __name__ == "__main__":` block lets you run it locally with:
  ```bash
  python simple_mcp_calculator.py
  ```

---
