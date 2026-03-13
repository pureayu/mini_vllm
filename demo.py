from mini_vllm.engine import Engine
from mini_vllm.request import Request


def main() -> None:
    engine = Engine(num_blocks=8, block_size=4, max_batch_size=2)

    requests = [
        Request(
            request_id=1,
            max_tokens=2,
            num_computed_tokens=0,
            prompt_tokens=[10, 11, 12],
        ),
        Request(
            request_id=2,
            max_tokens=3,
            num_computed_tokens=0,
            prompt_tokens=[20, 21],
        ),
    ]

    for request in requests:
        engine.add_request(request)

    engine.run()

    for request in engine.get_finished_requests():
        print(
            f"request_id={request.request_id}, "
            f"prompt_tokens={request.prompt_tokens}, "
            f"generated_tokens={request.generated_tokens}, "
            f"status={request.status}"
        )


if __name__ == "__main__":
    main()
