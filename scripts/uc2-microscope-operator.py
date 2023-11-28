import os
from typing import Any, Dict, List, Optional, Union
import asyncio
from schema_agents.role import Role
from schema_agents.schema import Message
from pydantic import BaseModel, Field
import re
from imjoy_rpc.hypha import connect_to_server


class MoveStageAction(BaseModel):
    """Move the stage on the microscope."""
    x: int = Field(description="Move the stage along X direction (um).")
    y: int = Field(description="Move the stage along Y direction (um).")
    z: int = Field(description="Move the stage along Z direction (um).")


class SnapImageAction(BaseModel):
    """Snap an image from microscope"""
    path: str = Field(description="File path to save the image.")


async def main():
    server = await connect_to_server({"server_url": "https://ai.imjoy.io/"})

    svc = await server.get_service("microscope-control")

    async def respond_to_user(query: str, role: Role = None) -> str:
        """Respond to user's request by generating microscope control data."""
        response = await role.aask(query, Union[MoveStageAction,SnapImageAction])
        if isinstance(response, MoveStageAction):
            await svc.move(value=response.x, axis="X", is_absolute=False, is_blocking=True)
            await svc.move(value=response.y, axis="Y", is_absolute=False, is_blocking=True)
        elif isinstance(response,SnapImageAction):
            await svc.getImage(path=response.path)
        return "done"
    microscope_operator = Role(
        name="MicroscopeOperator",
        profile="Microscope Controller",
        goal="Your goal is control the microscope based on the user's request.",
        constraints=None,
        actions=[respond_to_user],
    )
    event_bus = microscope_operator.get_event_bus()
    event_bus.register_default_events()

    # Simulate user query
    user_query = "Move the stage 1000 um through x axis."
    responses = await microscope_operator.handle(Message(content=user_query, role="User"))

    # Simulate user query
    # user_query = "Move the stage -2000 um along X direction, -2000 um along Y direction, 0 um along Z direction, and capture an image for 50 ms."
    # responses = await microscope_operator.handle(Message(content=user_query, role="User"))


if __name__ == "__main__":
    asyncio.run(main())



