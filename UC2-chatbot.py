import os

import asyncio
from schema_agents.role import Role
from schema_agents.schema import Message
from pydantic import BaseModel, Field
import re
from imjoy_rpc.hypha import connect_to_server




# async def init_service():
#     server = await connect_to_server({"server_url": "https://ai.imjoy.io/"})
    
#     svc = await server.get_service("microscope-control")
#     return svc




class MicroscopeControl(BaseModel):
    """Data for controlling the microscope."""
    x: int = Field(description="Move the stage along X direction (um).")
    y: int = Field(description="Move the stage along Y direction (um).")
    z: int = Field(description="Move the stage along Z direction (um).")
    ms: int = Field(description="Exposure time of the Camera (ms).")

async def respond_to_user(query: str, role: Role = None) -> MicroscopeControl:
    """Respond to user's request by generating microscope control data."""
    # Use a simple regex to extract numerical values from the user query
    values = [float(val) if '.' in val else int(val) for val in re.findall(r'\d+(?:\.\d*)?', query)]
    
    # Assign values to x, y, z, and ms, assuming a specific order in the query
    x, y, z, ms = values[:4]

    # Generate MicroscopeControl data
    response = MicroscopeControl(x=x, y=y, z=z, ms=ms)
    return response

async def main():
    microscope_operator = Role(
        name="MicroscopeOperator",
        profile="Microscope Controller",
        goal="Your goal is to listen to user's request and generate microscope control data.",
        constraints=None,
        actions=[respond_to_user],
    )
    event_bus = microscope_operator.get_event_bus()
    event_bus.register_default_events()

    # Simulate user query
    user_query = "Move the stage 1000 um along X direction, 200 um along Y direction, 0 um along Z direction, and capture an image for 50 ms."
    
    # Handle user query and print the generated microscope control data
    responses = await microscope_operator.handle(Message(content=user_query, role="User"))

    print(responses)
   
    #Connect to Hypha service registered by ImSwitch
    server = await connect_to_server({"server_url": "https://ai.imjoy.io/"})
    # Get an existing service
    # Since "hello-world" is registered as a public service, we can access it using only the name "hello-world"
    svc = await server.get_service("microscope-control")
    ret = await svc.move(value=int(responses["x"]), axis="X", is_absolute=False, is_blocking=True)
    print(ret)
    print("Stage moved!")
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()



