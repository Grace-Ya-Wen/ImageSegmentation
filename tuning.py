

def tuning(params):
    X_train, X_test, y_train, y_test = train_test_split(params['x'], params['y'],train_size=0.8,random_state=42)
    train_img, train_mask = image_processing(X_train, y_train, target_size = (128, 128), augmentation = True, padding = True)
    validate_img, validate_mask = image_processing(X_test, y_test, target_size = (128, 128), augmentation = True, padding = True)
    
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    input_img = Input((128, 128, 3), name='img')
    model = get_unet(input_img, n_filters = params['n_filters'], dropout = params['dropout'], batchnorm=True)
    model.compile(optimizer=Adam(lr = params['lr']), loss="binary_crossentropy", metrics=["accuracy"])
    #model.summary()
    model.fit(train_img, train_mask,batch_size=params['batch_size'], epochs=200,shuffle = True,callbacks=[earlystop], validation_data=(X_test,y_test))
    score, acc = model.evaluate(validate_img, validate_mask, verbose=0)
    return {'loss': score, 'status': STATUS_OK, 'model': model}
