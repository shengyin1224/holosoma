import os
import argparse

XML_TEMPLATE = """<mujoco model="{robot_type}_{object_name}">
    <!-- 1. 包含基础机器人模型 -->
    <include file="{robot_base_xml}"/>

    <!-- 2. 任务相关的资源定义 -->
    <asset>
        <mesh name="{object_name}_mesh" file="{mesh_path}" scale="{scale} {scale} {scale}"/>
    </asset>

    <!-- 3. 任务相关的物体定义 -->
    <worldbody>
        <body name="{object_name}_link" pos="0 0 0">
            <freejoint/>
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.002 0.002 0.002"/>
            <geom name="{object_name}" type="mesh" mesh="{object_name}_mesh" 
                  rgba="1.0 0.4235 0.0392 1.0" contype="1" conaffinity="1" 
                  friction="0.9 0.5 0.5" solref="0.02 1" solimp="0.9 0.95 0.001"/>
        </body>
        
        <!-- 可选：添加场景光照（基础模型中通常已有，此处作为补充） -->
        <light name="task_light" pos="0 0 5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    </worldbody>
</mujoco>
"""

def generate_xml(robot_type, object_name, mesh_path, output_path, scale=1.0):
    """
    生成一个基于模板的任务 XML 文件。
    
    Args:
        robot_type: 机器人类型 (如 'g1', 't1')
        object_name: 物体名称 (用于 MuJoCo geom/body 命名)
        mesh_path: 物体 .obj 文件的路径 (建议使用绝对路径)
        output_path: 生成的 XML 保存路径
        scale: 物体缩放比例
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # 自动寻找基础机器人模型的 XML 路径
    # 假设脚本在 holosoma/src/holosoma_retargeting/data_utils/
    # 基础模型在 holosoma/src/holosoma_retargeting/models/{robot}/{robot}_29dof.xml
    script_dir = os.path.dirname(os.path.abspath(__file__))
    robot_base_xml_abs = os.path.abspath(os.path.join(script_dir, "..", "models", robot_type, f"{robot_type}_29dof.xml"))
    
    if not os.path.exists(robot_base_xml_abs):
        print(f"Warning: Base robot XML not found at {robot_base_xml_abs}")
        # 尝试备选路径
        robot_base_xml_abs = os.path.abspath(os.path.join(script_dir, "..", "..", "..", "models", robot_type, f"{robot_type}_29dof.xml"))

    # 计算基础模型相对于输出 XML 的相对路径，方便 MuJoCo 加载
    try:
        robot_base_xml_rel = os.path.relpath(robot_base_xml_abs, output_dir)
    except ValueError:
        # 如果在不同驱动器（Windows），使用绝对路径
        robot_base_xml_rel = robot_base_xml_abs

    content = XML_TEMPLATE.format(
        robot_type=robot_type,
        task_name=object_name,
        robot_base_xml=robot_base_xml_rel,
        object_name=object_name,
        mesh_path=os.path.abspath(mesh_path),
        scale=scale
    )

    with open(output_path, "w") as f:
        f.write(content)
    
    print("-" * 50)
    print(f"✅ 任务 XML 已生成!")
    print(f"📍 输出路径: {os.path.abspath(output_path)}")
    print(f"🤖 基础机器人: {robot_base_xml_rel}")
    print(f"📦 物体模型: {os.path.abspath(mesh_path)}")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动生成 MuJoCo 任务场景 XML (包含机器人和物体)")
    parser.add_argument("--robot", type=str, default="g1", help="机器人类型，如 g1")
    parser.add_argument("--object", type=str, required=True, help="物体名称")
    parser.add_argument("--mesh", type=str, required=True, help="物体的 .obj 文件路径")
    parser.add_argument("--output", type=str, required=True, help="生成的 XML 保存路径")
    parser.add_argument("--scale", type=float, default=1.0, help="物体缩放比例")
    args = parser.parse_args()
    
    generate_xml(args.robot, args.object, args.mesh, args.output, args.scale)

