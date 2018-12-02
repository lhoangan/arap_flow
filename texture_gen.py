import bpy
import sys
import colorsys
import os, os.path as osp
import random as rn
from abc import ABC
from datetime import datetime as dt


# num : scene number, used to determine resource
def setup_scene(name):

    # create new scene
    bpy.ops.scene.new()
    scene = bpy.context.screen.scene

    # scene setup
    scene.name = name
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0

    #  delete the startup scene and all objects
    for o in bpy.data.objects:
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.scenes.remove(bpy.data.scenes['Scene'], do_unlink=True)

    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.feature_set = 'EXPERIMENTAL'

    scene.update()

def setup_render(render_root=''):

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'

    # rendering configuration
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = 10
    scene.render.tile_x = 256
    scene.render.tile_y = 256
    scene.render.threads_mode = 'FIXED'
    #scene.render.threads = 32
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.compression = 100
    #scene.render.use_multiview = True # stereo camera

    #renderlayer.setup(scene)
    #utils.generate_node_tree(renderlayer.get_composition_tree(render_root),
    #        scene.node_tree)



# setting up blender file
def init():

    bpy.ops.file.autopack_toggle()

    # setting view to orthographic projection
    # if run from text editor, the context isn't a 3d view, so bpy.ops.view3d.* 
    # will fail. So either make an operator and run from spacebar menu,
    # while the mouse is over 3d view, or use an override:
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = {'area': area, 'region': area.regions[-1]}
            bpy.ops.view3d.view_persportho(override)

class BData(ABC):

    @property
    def angles(self):
        return self._obj.rotation_euler

    @angles.setter
    def angles(self, val):
        self._obj.rotation_euler = val

    @property
    def location(self):
        return self._obj.location

    @location.setter
    def location(self, val):
        self._obj.location = val


class Light(BData):
    def __init__(self, name, dtype):

        assert dtype=='POINT' or dtype=='SUN', 'Data type not recognized: '+dtype
        assert name not in bpy.data.lamps, 'Data name occupied: '+name
        assert name not in bpy.data.objects, 'Object name occupied: '+name
        self._data = bpy.data.lamps.new(name, type=dtype)
        self._data.shadow_soft_size = rn.uniform(0, 5)
        self._data.use_nodes = True
        col = colorsys.hsv_to_rgb(rn.uniform(0, 1), min(max(0, rn.gauss(0.35, .25)), 1), 1)
        col = srgb_to_linear(*col)
        self.color = (col[0], col[1], col[2], 1)
        self._dtype = dtype
        self._obj = bpy.data.objects.new(name, self._data)
        bpy.context.scene.objects.link(self._obj)

    @property
    def intensity(self):
        return self._data.node_tree.nodes['Emission'].inputs['Strength'].default_value

    @intensity.setter
    def intensity(self, val):
        self._data.node_tree.nodes['Emission'].inputs['Strength'].default_value = val


    @property
    def color(self):
        return self._data.node_tree.nodes['Emission'].inputs['Color'].default_value

    @color.setter
    def color(self, val):
        self._data.node_tree.nodes['Emission'].inputs['Color'].default_value = val

class Camera(BData):
    def __init__(self, name):

        assert name not in bpy.data.cameras, 'Data name occupied: '+name
        assert name not in bpy.data.objects, 'Object name occupied: '+name
        self._data = bpy.data.cameras.new(name)
        self._obj = bpy.data.objects.new(name, self._data)
        bpy.context.scene.objects.link(self._obj)

def linear_to_srgb(r, g, b):
    def linear(c):
        a = .055
        if c <= .0031308:
            return 12.92 * c
        else:
            return (1+a) * c**(1/2.4) - a
    return tuple(linear(c) for c in (r, g, b))

def srgb_to_linear(r, g, b):
    def srgb(c):
        a = .055
        if c <= .04045:
            return c / 12.92
        else:
            return ((c+a) / (1+a)) ** 2.4
    return tuple(srgb(c) for c in (r, g, b))

# from here: https://blender.stackexchange.com/questions/80034/fix-hsv-to-rgb-conversion
# blender hsv works in srgb space, while RGB is defined in linear space
col = colorsys.hsv_to_rgb(.4, .8, 1)
# (0.19999999999999996, 1, 0.5200000000000002)
srgb_to_linear(*col)
# Values to be set in the RGB tab of the color picker:
# (0.03310476657088504, 1.0, 0.23302199930143835)


def random_color():

    col = colorsys.hsv_to_rgb(rn.uniform(0, 1), rn.uniform(0, 1), 1)
    col = srgb_to_linear(*col)

    ## pick a random color
    #R = G = B = 0
    #while abs(R - G) < 1e-5 and abs(G - B) < 1e-5 and abs(R - B) < 1e-5:
    #    R = rn.uniform(0, 1)
    #    G = rn.uniform(0, 1)
    #    B = rn.uniform(0, 1)

    return (col[0], col[1], col[2], 1)


class BrickTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexBrick')
        texture.offset = rn.uniform(0, 1)
        texture.offset_frequency = rn.randint(1, 99)
        texture.squash = rn.uniform(0, 99)
        texture.squash_frequency = rn.randint(1, 99)
        texture.inputs[1].default_value = random_color()
        texture.inputs[2].default_value = random_color()
        texture.inputs[3].default_value = random_color()
        texture.inputs[4].default_value = rn.uniform(5, 70) # scale
        texture.inputs[5].default_value = rn.uniform(0.02, 0.08) # mortar size
        return texture

class CheckerTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexChecker')
        texture.inputs[1].default_value = random_color()
        texture.inputs[2].default_value = random_color()
        texture.inputs[3].default_value = rn.uniform(15, 100) # scale
        return texture

class MagicTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexMagic')
        texture.turbulence_depth = rn.randint(0, 10)
        texture.inputs[1].default_value = rn.uniform(15, 50)
        return texture


class MusgraveTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexMusgrave')
        texture.musgrave_type = rn.choice(['RIDGED_MULTIFRACTAL', 'FBM'])
        texture.inputs[1].default_value = rn.uniform(15, 100)
        return texture

class NoiseTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexNoise')
        texture.inputs[1].default_value = rn.uniform(15, 50)
        return texture

class VoronoiTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexVoronoi')
        texture.coloring = rn.choice(['CELLS', 'INTENSITY'])
        texture.inputs[1].default_value = rn.uniform(5, 100)
        return texture

class WaveTex():
    def __init__(self):
        pass
    def generate(self, nodes):
        texture = nodes.new(type='ShaderNodeTexWave')
        texture.wave_type = rn.choice(['RINGS', 'BANDS'])
        texture.wave_profile = rn.choice(['SAW', 'SIN'])
        texture.inputs[1].default_value = rn.uniform(5, 70)
        texture.inputs[2].default_value = rn.uniform(5, 100)
        return texture


def material_gen():

    names = ['Brick', 'Checker', 'Magic', 'Musgrave', 'Noise', 'Voronoi', 'Wave']
    textures = ['BrickTex()', 'CheckerTex()', 'MagicTex()', 'MusgraveTex()',
            'NoiseTex()', 'VoronoiTex()', 'WaveTex()']
    # picking a texture
    dice = rn.randint(1, 7)
    mat = bpy.data.materials.new(name=names[dice-1]+'_'+str(dt.now()))

    # turniing on node
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    texture = eval(textures[dice-1])
    tex = texture.generate(nodes)

    diffuse = nodes['Diffuse BSDF']
    diffuse.inputs[1].default_value = rn.uniform(0, 1)

    glossy = nodes.new(type='ShaderNodeBsdfGlossy')
    glossy.inputs[1].default_value = rn.uniform(0.40, 1.0) # glossy roughness

    mix = nodes.new(type='ShaderNodeMixShader')
    mix.inputs[0].default_value = max(0, min(rn.gauss(0.5, .25), .75)) # diffuse-glossy factor: 0 for all diffuse.

    links.new(tex.outputs[0], diffuse.inputs[0])
    links.new(diffuse.outputs[0], mix.inputs[1])
    links.new(glossy.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], nodes['Material Output'].inputs[0])

    return mat


# read input arguments

argv = sys.argv
try:
    argv = argv[argv.index('--') + 1:] # get all args after --
    fr = argv[0] # from number
    to = argv[1] # to number
except ValueError:
    print ('Input error!')

# creating a scene, named Gardyn
init()
setup_scene('Texture')
setup_render()

# creating a light source
lam = Light('lamp', 'POINT')
# create a camera
cam = Camera('camera')
cam.location = (0, 0, 3)

# creating a plane where the texture is
bpy.context.scene.cursor_location = (0.0, 0.0, 0.0)
bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0), radius=2)
plane = bpy.context.scene.objects.active
plane.name = 'plane'

for i in range(int(fr), int(to)):
    mat = material_gen()
    if len(plane.material_slots) == 0:
        plane.data.materials.append(mat)
    else:
        plane.material_slots[0].material = mat
    lam.location = (rn.uniform(-3, 3), rn.uniform(-3, 3), rn.uniform(3, 5))
    lam.intensity = rn.randint(500, 1000)
    col = colorsys.hsv_to_rgb(rn.uniform(0, 1), min(max(0, rn.gauss(0.35, .25)), 1), 1)
    col = srgb_to_linear(*col)
    lam.color = (col[0], col[1], col[2], 1)

    # rendering
    bpy.context.scene.camera = cam._obj
    bpy.context.scene.render.filepath = osp.join('/home/hale/tempt/{:05d}'.format(i))
    bpy.ops.render.render(write_still=True)


