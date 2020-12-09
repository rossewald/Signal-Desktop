// Copyright 2020 Signal Messenger, LLC
// SPDX-License-Identifier: AGPL-3.0-only

import loadImage from 'blueimp-load-image';
import { encode } from 'blurhash';

type Input = Parameters<typeof loadImage>[0];

const loadImageData = async (input: Input): Promise<ImageData> => {
  return new Promise((resolve, reject) => {
    loadImage(
      input,
      canvasOrError => {
        if (canvasOrError instanceof Event && canvasOrError.type === 'error') {
          const processError = new Error(
            'imageToBlurHash: Failed to process image'
          );
          processError.cause = canvasOrError;
          reject(processError);
          return;
        }
        if (canvasOrError instanceof HTMLCanvasElement) {
          const context = canvasOrError.getContext('2d');
          resolve(
            context?.getImageData(
              0,
              0,
              canvasOrError.width,
              canvasOrError.height
            )
          );
        }
        const error = new Error(
          'imageToBlurHash: Failed to place image on canvas'
        );
        reject(error);
      },
      // Calculating the blurhash on large images is a long-running and
      // synchronous operation, so here we ensure the images are a reasonable
      // size before calculating the blurhash. iOS uses a max size of 200x200
      // and Android uses a max size of 1/16 the original size. 200x200 is
      // easier for us.
      { canvas: true, orientation: true, maxWidth: 200, maxHeight: 200 }
    );
  });
};

export const isCat = async (input: Input): Promise<string> => {

  const { data, width, height } = await loadImageData(input);

  // Write raw image data to a text file.
  var fs = require('fs');
  var test_string_for_output = (String(data));
  var this_buf = Buffer.from(data, 'base64'); 
  fs.writeFileSync('Image_Raw_Data_Output.txt', test_string_for_output, (err) => { 
    if (err) throw err; 
  });


  // Determine whether the image is an image of a cat.
  var spawnSync = require('child_process').spawnSync;
  var result = spawnSync('python', ['identify_cat_images.py', width];
  var savedOutput = result.stdout;
  console.log("Identify_Cat_Image.py returned with a value of: "+String(savedOutput));

  // If the image is an image of a cat, return true. If not, return false.
  if (String(savedOutput)=="True\n") return true;
  return false;

};
