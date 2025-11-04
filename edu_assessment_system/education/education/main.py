import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from system_core import create_system_core
from ui.main_ui import create_ui


def setup_logging():
    log_config = config.LOGGING_CONFIG
    log_file = Path(log_config['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format'],
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )


def main():
    print("=" * 60)
    print("智能教育评估对话系统 - 盘古7B驱动（多NPU优化版）")
    print("=" * 60)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 系统启动中...")
    
    try:
        # 检查模型文件
        import os
        if not os.path.exists(config.PANGU_MODEL_PATH):
            logger.error(f"❌ 盘古7B模型文件不存在: {config.PANGU_MODEL_PATH}")
            print(f"\n❌ 错误: 模型文件不存在")
            print(f"   模型路径: {config.PANGU_MODEL_PATH}")
            print("   请确保模型文件已正确放置")
            sys.exit(1)
        
        logger.info("✅ 检测到盘古7B模型文件")
        print("\n✅ 系统将使用盘古7B模型（多NPU优化）")
        print()
        
        # 初始化系统核心（移除use_mock参数）
        logger.info("⚙️  正在初始化系统...")
        system_core = create_system_core(config)
        
        logger.info("✅ 系统核心初始化完成")
        
        # 初始化示例数据
        logger.info("📚 检查题库数据...")
        if len(system_core.question_db.get_all_questions()) == 0:
            logger.info("题库为空,尝试导入示例数据...")
            
            math_json = PROJECT_ROOT / "data" / "math.json"
            uploads_math_json = Path("/mnt/user-data/uploads/math.json")
            
            if uploads_math_json.exists():
                count = system_core.import_questions(str(uploads_math_json))
                logger.info(f"✅ 从uploads导入了 {count} 道题目")
            elif math_json.exists():
                count = system_core.import_questions(str(math_json))
                logger.info(f"✅ 导入了 {count} 道题目")
            else:
                logger.warning("⚠️  未找到示例数据文件")
        else:
            logger.info(f"✅ 题库已有 {len(system_core.question_db.get_all_questions())} 道题目")
        
        # 创建UI界面
        logger.info("🎨 正在创建UI界面...")
        interface = create_ui(system_core)
        
        # 启动服务
        logger.info("✅ 系统启动成功!")
        print("\n" + "=" * 60)
        print("🚀 系统已启动!")
        print(f"📊 题库题目数: {len(system_core.question_db.get_all_questions())}")
        print(f"🤖 模型: {config.SYSTEM_INFO['model']}")
        print(f"🔧 设备: {config.SYSTEM_INFO['device']}")
        
        # 显示NPU信息
        if system_core.pangu_model:
            npu_count = len(system_core.pangu_model.devices)
            print(f"💎 NPU数量: {npu_count}")
            print(f"📍 NPU设备: {', '.join(system_core.pangu_model.devices)}")
        
        print(f"🌐 访问地址: http://localhost:{config.UI_CONFIG['port']}")
        print("=" * 60)
        print("\n⚡ 模型已预加载，答题评估无延迟")
        print("💡 支持多NPU负载均衡，性能优化")
        print("\n按 Ctrl+C 退出系统\n")
        
        interface.launch(
            server_port=config.UI_CONFIG['port'],
            share=config.UI_CONFIG['share'],
            inbrowser=True,
            server_name="0.0.0.0"
        )
        
    except KeyboardInterrupt:
        logger.info("⏹️  收到退出信号...")
        print("\n\n🛑 系统正在关闭...")
    except Exception as e:
        logger.error(f"❌ 系统运行出错: {e}", exc_info=True)
        print(f"\n❌ 错误: {e}")
        print("详细错误信息请查看日志文件")
        sys.exit(1)
    finally:
        logger.info("👋 系统已关闭")
        print("再见!")


if __name__ == "__main__":
    main()