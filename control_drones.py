from mavsdk import System
import asyncio
from mavsdk.offboard import OffboardError, PositionNedYaw

async def connect_drone(mavsdk_server_address, port):
    """Connects a drone to the system and waits for global position lock."""
    drone = System(mavsdk_server_address=mavsdk_server_address, port=port)
    print(f"Jungiamasi prie {mavsdk_server_address}:{port}...")
    await drone.connect()

    # Wait for connection
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Dronas prisijungė prie {mavsdk_server_address}:{port}")
            break

    # Wait for a valid global position before arming
    print("Waiting for valid global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print(" Global position is OK.")
            break

    return drone

async def arm_and_takeoff(drone):
    """Arm'ina droną ir pakelia"""
    print(f"Ruošiamas dronas {drone}:")
    await drone.action.arm()
    await asyncio.sleep(2)  # Šiek tai laiko paruošti

    print(f"Kyla dronas {drone}:")
    await drone.action.takeoff()

    # Wait for takeoff and stabilize
    await asyncio.sleep(11)  # Šiek tiek laiko stabilizuotis po pakilimo

async def activate_offboard(drone):
    """Aktyvuoja offboard režimą ir pastoviai siunčia pozicijos atnaujinimus."""
    print(f"Aktyvuojamas offboard režimas dronui - {drone}...")

    # Gaunam drono dabartine pozicija pries aktyvuojant Offboard rezima
    async for position in drone.telemetry.position_velocity_ned():
        current_x = position.position.north_m  # Get current X
        current_y = position.position.east_m   # Get current Y
        current_z = position.position.down_m   # Get current altitude
        break  # Exit loop after getting the first reading

    # pradinė pozicija
    initial_position = PositionNedYaw(current_x, current_y, current_z, 0.0)

    try:
        print(f" dabartinę poziciją - {initial_position} nustatom kaip pradinį nustatytą tašką...")

        #  Send the setpoint multiple times to overwrite any previous Offboard state
        for _ in range(10):  
            await drone.offboard.set_position_ned(initial_position)
            await asyncio.sleep(0.1)

        # Startuojam Offboard režimą
        await drone.offboard.start()
        print(" Offboard režimas aktyvuotas.")
        await asyncio.sleep(3)

    except OffboardError as e:
        print(f" Klaida aktyvuojant offboard režimą: {e}")
        return False

    # **Siunčiam poziciją, kad išvengti offboard išsijungimo**
    print(" Palaikomas Offboard režimas...")
    try:
        for _ in range(10):  # Keep sending updates for 10 seconds
            await drone.offboard.set_position_ned(initial_position)
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f" Klaida palaikant offboard režimą: {e}")

    return True

async def fly_drone_forward(drone, x, y, z, yaw):
    """Skraidina droną pastoviai siunčiant pozicijos atnaujinimus."""
    print(f" Dronas - {drone} - skrenda į - x:{x}, y:{y}, z:{z} (yaw: {yaw})")

    position = PositionNedYaw(x, y, z, yaw)

    try:
        for _ in range(10):  # Move for 10 seconds (5Hz updates)
            await drone.offboard.set_position_ned(position)
            await asyncio.sleep(0.2)
    except OffboardError as e:
        print(f" Klaida skrendant: {e}")

async def fly_square(drone):
    """Skraidinam droną kvadratiniu šablonu."""

    # #  grįžta į pradinę poziciją, jei ne visi ten yra.
    # await fly_drone_forward(drone, 0.0, 0.0, -5.0, 0.0)
    # await asyncio.sleep(5)  # Šiek tiek laiko atlikti manevrą

    await fly_drone_forward(drone, 5.0, 10.0, -5.0, 60.0)  # Skraidinam į pirmą kampą
    await asyncio.sleep(5)  # Šiek tiek laiko atlikti manevrą

    await fly_drone_forward(drone, 0.0, 20.0, -5.0, 120.0)  # Skraidinam į antrą kampą
    await asyncio.sleep(5) # Šiek tiek laiko atlikti manevrą

    await fly_drone_forward(drone, -5.0, 10.0, -5.0, -120.0)  # Skraidianam į trečią kampą
    await asyncio.sleep(5) # Šiek tiek laiko atlikti manevrą

    await fly_drone_forward(drone, 0.0, 0.0, -5.0, -60.0)  # Grąžinam į pradinę poziciją.
    await asyncio.sleep(5) # Šiek tiek laiko atlikti manevrą

    print(f"Dronas -  {drone} - užbaigė skrydį formos šablonu!")


async def land_drone(drone):
    """Nuleidžia droną."""
    print(f" Leidžiamas dronas - {drone}:")
    await drone.action.land()
    await asyncio.sleep(10)  # Šiek tiek laiko dronui nusileisti

async def stop_offboard_mode_for_all_drones(drones):
    """Sustabdom offboard režimą visiems dronams."""
    await asyncio.gather(*(stop_offboard_mode(drone) for drone in drones))

async def stop_offboard_mode(drone):
    """Sustabdom offboard režimą vienam dronui"""
    try:
        print(f"Stabdomas offboard režimas dronui - {drone}...")
        await drone.offboard.stop()  # Attempt to stop Offboard mode
        print(f"Offboard režimas sėkmingai sustabdytas dronui - {drone}.")
    except Exception as e:
        print(f"Klaida išjungiant offboard dronui - {drone}: {e}")

async def control_drones():
    """Main function to control multiple drones."""
    drones_info = [
        ("localhost", 50060),  # Drone 1
        ("localhost", 50061),  # Drone 2
        ("localhost", 50062),  # Drone 3
    ]

    drones = [await connect_drone(address, port) for address, port in drones_info]

    # Paruošiam ir pakeliam visus dronus
    await asyncio.gather(*(arm_and_takeoff(drone) for drone in drones))

    await asyncio.sleep(5)  # Šiek tiek laiko stabilizuotis

    # Aktyvuojam Offboard visiems dronams
    await asyncio.gather(*(activate_offboard(drone) for drone in drones))

    await asyncio.sleep(2)  # Šiek tiek laiko prieš pradedant judėjimą

    # Skraidinam dronus
    await asyncio.gather(
        # *(fly_drone_forward(drone, 0.0, 0.0, -5.0, 90.0) for drone in drones)
        *(fly_square(drone) for drone in drones)
    )

    await asyncio.sleep(2)  # šiek tiek laiko stabilizuotis

    await asyncio.gather(
        fly_drone_forward(drones[0], 15.0, 0.0, -5.0, 0.0),
        fly_drone_forward(drones[1], -10.0, 20.0, -5.0, 60.0),
        fly_drone_forward(drones[2], 5.0, 20.0, -5.0, 30.0),
    )

    
    await asyncio.sleep(2)  # šiek tiek laiko stabilizuotis

    # Nuleidžiam visus dronus
    await asyncio.gather(*(land_drone(drone) for drone in drones))

    await asyncio.sleep(7) # šiek tiek laiko nusileidimui.

    # Išjungiam Offboard visiems dronams
    await stop_offboard_mode_for_all_drones(drones)
        

if __name__ == "__main__":
    asyncio.run(control_drones())

