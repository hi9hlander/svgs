from panda3d.core import *
from direct.interval.IntervalGlobal import *
from odvm.renderer import Renderer
from odvm.optimizedvm import OptimizedVM
from random import randrange
import cProfile
import pstats


viewport_vert_glsl = """ 
#version 130

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat3 p3d_NormalMatrix;
uniform vec3 gnormal;

in vec4 p3d_Vertex;
in vec4 p3d_Color;
in vec2 p3d_MultiTexCoord0;

     out vec2  texcoord;
flat out vec4  color;
flat out vec2  pnormal;
     out float pos_w;

void main() 
{ 
   vec4    pos = p3d_ModelViewProjectionMatrix*p3d_Vertex;
   gl_Position = pos;
   texcoord    = p3d_MultiTexCoord0;
   color       = p3d_Color;
   vec3 nrm    = p3d_NormalMatrix*gnormal;
   pnormal     = nrm.xy*inversesqrt(0.5*nrm.z+0.5);
   pos_w       = 0.01*pos.w;
}
""" 

viewport_frag_glsl = """ 
#version 130

uniform sampler2D p3d_Texture0; 

     in vec2  texcoord;
flat in vec4  color;
flat in vec2  pnormal;
     in float pos_w;

void main() 
{ 
   vec4 clr = texture2D(p3d_Texture0,texcoord)*color;
   gl_FragData[0] = clr;
   gl_FragData[1] = vec4( pnormal, pos_w, step(0.5,clr.a) );
} 
""" 

composer_vert_glsl = """ 
#version 130

out vec2 texcoord;

void main()
{
   vec2     ij = vec2(float((gl_VertexID&1)<<2),float((gl_VertexID&2)<<1));
   gl_Position = vec4(ij-1.0,0.0,1.0);
   texcoord    = ij*0.5;
} 
""" 

composer_frag_glsl = """ 
#version 130

uniform sampler2D p3d_Texture0; 
uniform sampler2D aux0;
uniform vec4      viewport;
uniform float     thickness;

in vec2 texcoord;

void main() 
{ 
   vec4    clr = texture2D(p3d_Texture0,texcoord);
   vec3 nxy_pz = texture2D(aux0,texcoord).xyz;
   float  x2y2 = dot(nxy_pz.xy,nxy_pz.xy);
   vec3    nrm = vec3( nxy_pz.xy*sqrt(1.0-0.25*x2y2), 1.0-0.5*x2y2 );
   vec3    eye = normalize(vec3(gl_FragCoord.xy*viewport.st+viewport.pq,1.0));
   float   dne = max(dot(nrm,eye),0.0);
   float   pti = dne*0.5;
   vec3    dtl = vec3(0.0,1.0,0.0);
   float   dti = max(dot(nrm,dtl)*0.5,0.0);
   float   fni = pti+dti+0.2;

   float  ds = thickness*dFdx(texcoord.s);
   float  dt = thickness*dFdy(texcoord.t);

   float z01 = texture2D(aux0,vec2(texcoord.s   ,texcoord.t-dt)).z;
   float z10 = texture2D(aux0,vec2(texcoord.s-ds,texcoord.t   )).z;
   float z12 = texture2D(aux0,vec2(texcoord.s+ds,texcoord.t   )).z;
   float z21 = texture2D(aux0,vec2(texcoord.s   ,texcoord.t+dt)).z;

   float e   = abs(nxy_pz.z-max(max(z01,z21),max(z10,z12)))*dne;
   float g   = step(0.01*nxy_pz.z,e);

   gl_FragColor = clr*fni*(1.0-g);
} 
""" 


def unlock_camera(): base.disable_mouse()


def lock_camera():
   mat = Mat4(base.camera.get_mat())
   mat.invert_in_place()
   base.mouseInterfaceNode.set_mat(mat)
   base.enable_mouse()


class Kubics(Renderer):
   def __init__(self):
      Renderer.__init__(self)
      self.viewport.add_shader( viewport_vert_glsl, viewport_frag_glsl )
      self.composer.add_shader( composer_vert_glsl, composer_frag_glsl )
      base.set_frame_rate_meter(True)

      self.model = OptimizedVM( 'VoxelModel', GeomVertexFormat.get_v3c4(), 1 )
      self.model.set_shader_input( Vec3( 1.0,0.0,0.0), 'gnormal', Vec3( 1.0,0.0,0.0) )
      self.model.set_shader_input( Vec3(-1.0,0.0,0.0), 'gnormal', Vec3(-1.0,0.0,0.0) )
      self.model.set_shader_input( Vec3(0.0, 1.0,0.0), 'gnormal', Vec3(0.0, 1.0,0.0) )
      self.model.set_shader_input( Vec3(0.0,-1.0,0.0), 'gnormal', Vec3(0.0,-1.0,0.0) )
      self.model.set_shader_input( Vec3(0.0,0.0, 1.0), 'gnormal', Vec3(0.0,0.0, 1.0) )
      self.model.set_shader_input( Vec3(0.0,0.0,-1.0), 'gnormal', Vec3(0.0,0.0,-1.0) )

      self.model_path = render.attach_new_node(self.model)
      self.model_path.set_color_off()
      self.model_path.set_attrib(CullFaceAttrib.make(CullFaceAttrib.MCullClockwise))
      self.model_path.set_transparency(TransparencyAttrib.MDual)
      self.model_path.set_render_mode_filled()
      self.composer.output.set_shader_input( 'thickness', 2.0 )
      
      unlock_camera()
      base.camera.set_pos(0,16,32)
      base.camera.look_at(0,0,0)
      lock_camera()

      base.accept( 'w', self.toggle_wireframe )

      base.bufferViewer.position = 'llcorner'
      base.bufferViewer.setCardSize(0,0.5)
      base.bufferViewer.layout   = 'vline'

      base.accept( 'v', self.toggle_cards )

   def toggle_wireframe(self):
      if   self.model_path.get_render_mode() == 2:
         self.model_path.set_render_mode_filled()
         self.composer.output.set_shader_input( 'thickness', 2.0 )
      elif self.model_path.get_render_mode() == 1 or self.model_path.get_render_mode() == 0:
         self.model_path.set_render_mode_wireframe()
         self.composer.output.set_shader_input( 'thickness', 0.0 )

   def toggle_cards(self):
      render.analyze()
      print( self.viewport.depth_stencil, self.viewport.color )
      if hasattr(self.viewport,'aux0'): print( self.viewport.aux0 )
      base.bufferViewer.toggleEnable()

   def init_kubics( self, di=8, dj=16 ):
      self.kcolor  = Vec4(1.0,1.0,1.0,1.0)
      self.kpoints = []
      self.di      = di
      self.dj      = dj
      self.field   = [ [-1]*self.di for j in range(self.dj) ]

      self.seq_down = Sequence(Wait(0.5),Func(self.move_down))
      self.seq_down.loop(0.0,-1.0,1.0)

      base.accept( 'arrow_left'        , self.move_left    )
      base.accept( 'arrow_left-repeat' , self.move_left    )
      base.accept( 'arrow_right'       , self.move_right   )
      base.accept( 'arrow_right-repeat', self.move_right   )
      base.accept( 'home'              , self.rotate_left  )
      base.accept( 'home-repeat'       , self.rotate_left  )
      base.accept( 'page_up'           , self.rotate_right )
      base.accept( 'page_up-repeat'    , self.rotate_right )
      base.accept( 'arrow_up'          , self.rotate_left  )
      base.accept( 'arrow_up-repeat'   , self.rotate_left  )
      base.accept( 'arrow_down'        , self.drop_down    )
      base.accept( 'arrow_down-repeat' , self.drop_down    )

   def draw_field(self):
      with self.model.quads:
         for i in range(-(self.di>>1)-1,(self.di>>1)+1): self.model.add(0,i,-8,0,Vec4(1.0,1.0,1.0,1.0))
         for j in range(self.dj,0,-1):
            self.model.add(0,-(self.di>>1)-1,j-(self.dj>>1),0,Vec4(1.0,1.0,1.0,1.0))
            self.model.add(0, (self.di>>1)  ,j-(self.dj>>1),0,Vec4(1.0,1.0,1.0,1.0))
         self.model.optimize()

   kubics = ( ( Vec4(1.0,1.0,1.0,1.0), (  0, 2,  0, 0,  0, -2, -2,  2 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), (  0, 2,  0, 0,  0, -2,  2,  2 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), (  0, 2,  0, 0,  0, -2,  2,  0 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), (  0, 2,  0, 0,  0, -2,  2,  0 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), (  1, 2,  1, 0, -1,  0, -1, -2 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), ( -1, 2, -1, 0,  1,  0,  1, -2 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), (  0, 2,  0, 0,  0, -2,  0, -4 ) ),
              ( Vec4(1.0,1.0,1.0,1.0), ( -1, 1,  1, 1, -1, -1,  1, -1 ) ) )

   def select_kubics( self, idx ):
      self.kidx   = idx
      self.kcolor = Kubics.kubics[self.kidx][0]
      self.ki     = self.di>>1
      self.kj     = self.dj-2
      del self.kpoints[:]
      for p in range(0,len(Kubics.kubics[self.kidx][1]),2): self.kpoints.append( [Kubics.kubics[self.kidx][1][p],Kubics.kubics[self.kidx][1][p+1]] )

   def rotate_kubics( self, si, sj ):
      for p in self.kpoints: p[0],p[1] = si*p[1],sj*p[0]

   def draw_kubics(self):
      i = self.ki-(self.di>>1)
      j = self.kj-(self.dj>>1)+1
      with self.model.quads:
         for p in self.kpoints: self.model.add(0,i+(p[0]>>1),j+(p[1]>>1),0,self.kcolor)
         self.model.optimize()

   def erase_kubics(self):
      i = self.ki-(self.di>>1)
      j = self.kj-(self.dj>>1)+1
      with self.model.quads:
         for p in self.kpoints: self.model.sub(0,i+(p[0]>>1),j+(p[1]>>1),0)
         self.model.optimize()

   def check_fit( self, ki, kj, si=0, sj=0 ):
      for p in self.kpoints:
         if si == 0:
            i = ki+(p[0]>>1)
            j = kj+(p[1]>>1)
         else:
            i = ki+((si*p[1])>>1)
            j = kj+((sj*p[0])>>1)
         if i < 0 or i >= self.di or j < 0 or j >= self.dj or self.field[j][i] != -1: return False
      else: return True

   def rotate_left(self):
      if not self.check_fit( self.ki, self.kj, -1, 1 ): return False
      with self.model.quads:
         self.erase_kubics()
         self.rotate_kubics(-1,1)
         self.draw_kubics()
      return True

   def rotate_right(self):
      if not self.check_fit( self.ki, self.kj, 1, -1 ): return False
      with self.model.quads:
         self.erase_kubics()
         self.rotate_kubics(1,-1)
         self.draw_kubics()
      return True

   def move_left(self):
      if not self.check_fit( self.ki-1, self.kj ): return False
      with self.model.quads:
         self.erase_kubics()
         self.ki -= 1
         self.draw_kubics()
      return True

   def move_right(self):
      if not self.check_fit( self.ki+1, self.kj ): return False
      with self.model.quads:
         self.erase_kubics()
         self.ki += 1
         self.draw_kubics()
      return True

   def move_down(self):
      if not self.check_fit( self.ki, self.kj-1 ):
         rows = set()
         for p in self.kpoints:
            i = self.ki+(p[0]>>1)
            j = self.kj+(p[1]>>1)
            self.field[j][i] = self.kidx
            rows.add(j)
         rebuild_rqrd = False
         for j in sorted(rows,reverse=True):
            if -1 not in self.field[j]:
               d = self.field[j]
               for m in range(j+1,len(self.field)): self.field[m-1] = self.field[m]
               self.field[-1] = d
               for i in range(len(self.field[-1])): self.field[-1][i] = -1
               rebuild_rqrd = True
         if rebuild_rqrd:
            with self.model.quads:
               self.model.set_opt_level(2)
               for kj in range(len(self.field)):
                  for ki in range(len(self.field[j])):
                     i = ki-(self.di>>1)
                     j = kj-(self.dj>>1)+1
                     self.model.sub(0,i,j,0)
                     kidx = self.field[kj][ki]
                     if kidx != -1: self.model.add(0,i,j,0,Kubics.kubics[kidx][0])
               self.model.optimize()
               self.model.set_opt_level(1)
         self.select_kubics(randrange(len(Kubics.kubics)))
         if not self.check_fit( self.ki, self.kj ):
            self.toggle_wireframe()
            self.seq_down.finish()
         else:
            self.draw_kubics()
            self.seq_down.loop(0.0,-1.0,1.0)
         return False
      with self.model.quads:
         self.erase_kubics()
         self.kj -= 1
         self.draw_kubics()
      return True

   def drop_down(self): self.seq_down.loop(0.0,-1.0,10.0)

game = Kubics()
game.init_kubics()
game.draw_field()
game.select_kubics(randrange(len(Kubics.kubics)))
game.draw_kubics()

cProfile.run('game.run()','kubics.profile')
pstats.Stats('kubics.profile').strip_dirs().sort_stats('time').print_stats()
