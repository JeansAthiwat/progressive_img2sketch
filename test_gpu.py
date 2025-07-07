# import bpy

# # Make sure Cycles addon is enabled
# if "cycles" not in bpy.context.preferences.addons:
#     bpy.ops.preferences.addon_enable(module="cycles")

# # Set Cycles as the render engine
# bpy.context.scene.render.engine = 'CYCLES'

# # Set device type
# prefs = bpy.context.preferences.addons['cycles'].preferences
# prefs.compute_device_type = 'CUDA'  # Change to 'HIP', 'OPTIX' as needed

# # Get and enable all available GPU devices
# prefs.get_devices()
# print("Available Devices:")
# for device in prefs.devices:
#     print(f"  {device.name} - {device.type} - {'Enabled' if device.use else 'Disabled'}")
#     device.use = True  # Enable device

# # Set scene to use GPU
# bpy.context.scene.cycles.device = 'GPU'

import bpy

# Enable Cycles addon
if "cycles" not in bpy.context.preferences.addons:
    bpy.ops.preferences.addon_enable(module="cycles")

bpy.context.scene.render.engine = "CYCLES"

# Set GPU device
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'  # Or 'OPTIX'
prefs.get_devices()
for d in prefs.devices:
    d.use = True
bpy.context.scene.cycles.device = 'GPU'

# Create a test scene
bpy.ops.mesh.primitive_monkey_add()
bpy.ops.object.shade_smooth()

# Set output settings
bpy.context.scene.render.filepath = "/tmp/monkey_gpu.png"
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.image_settings.file_format = 'PNG'

# Render image
bpy.ops.render.render(write_still=True)
