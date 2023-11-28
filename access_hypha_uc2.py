import asyncio
from imjoy_rpc.hypha import connect_to_server

async def main():
    server = await connect_to_server({"server_url": "https://ai.imjoy.io/"})

    # Get an existing service
    # Since "microscope-control" is registered as a public service, we can access it using only the name "hello-world"
    svc = await server.get_service("microscope-control")
    await svc.move(value=0, axis="X", is_absolute=False, is_blocking=True)
    await svc.
if __name__ == "__main__":
    asyncio.run(main())

