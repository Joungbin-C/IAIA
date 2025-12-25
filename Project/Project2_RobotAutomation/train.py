from ultralytics import YOLO
import torch


def train_yolo_model():

    if torch.cuda.is_available():
        device = 0
        print("✅ CUDA is available. Using GPU.")
    else:
        device = 'cpu'
        print("⚠️ CUDA is not available. Using CPU.")

    model = YOLO('yolov8s.pt')

    try:
        results = model.train(
            data=r'C:/Users/joung/Downloads/spinnaker_python-4.2.0.88-cp310-cp310-win_amd64/train/surface-detecting-5/data.yaml',

            # ----------------------------
            # 학습 기본 설정
            # ----------------------------
            epochs=100,          # 100 → 150 으로 증가 (소형 결함용)
            imgsz=640,          # 작은 결함 검출 강화
            batch=8,             # VRAM 고려 (안전값)
            device=device,
            patience=20,
            workers=8,
            cache=True,

            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            weight_decay=0.0005,

            mosaic=0.0,
            mixup=0.0,
            flipud=0.0,

            project=r'C:/Users/joung/Downloads/spinnaker_python-4.2.0.88-cp310-cp310-win_amd64/train/runs',
            name='yolov8s_surface_defect_v1',
        )

        print("\n🎉 학습이 성공적으로 완료되었습니다!")
        print(f"📁 결과 저장 위치: {results.save_dir}")

    except Exception as e:
        print(f"\n❌ 학습 중 오류 발생: {e}")


if __name__ == '__main__':
    train_yolo_model()
