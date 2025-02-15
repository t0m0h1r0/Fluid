from .test_field.field_validator import FieldValidator


def main():
    """検証ツールのメインエントリーポイント"""
    validator = FieldValidator(
        shape=(32, 32, 32),  # デフォルトのグリッドサイズ
        rtol=1e-5,  # 相対許容誤差
        atol=1e-8,  # 絶対許容誤差
    )
    result = validator.run_validation_suite()

    if result:
        print("\n🎉 全ての検証テストに合格しました！")
        exit(0)
    else:
        print("\n❌ 一部または全ての検証テストで問題が見つかりました。")
        exit(1)


if __name__ == "__main__":
    main()
