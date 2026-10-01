import json
from pathlib import Path
# Import FastAPI application factory function to generate OpenAPI schema
from pynvr.app import create_app 

def export_schema():
    # Force FastAPI to resolve  Pydantic routes into a raw OpenAPI layout
    schema = create_app(system_config={}, nvr=None).openapi()
    
    # Save to local json file
    output_path = Path("./openapi.json")
    output_path.write_text(json.dumps(schema, indent=2))
    print(f"successfully wrote OpenAPI schema to {output_path.absolute()}")

if __name__ == "__main__":
    export_schema()