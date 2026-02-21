from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, epoch=None):
    model.train()
    total_loss = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}" if epoch is not None else "Training",
        leave=False
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({
            "batch_loss": f"{loss.item():.4f}"
        })

    return total_loss / len(loader)