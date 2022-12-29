import NextLink from "next/link";
import {
  Box,
  Heading,
  Text,
  Container,
  Button,
  Center,
  useColorModeValue,
} from "@chakra-ui/react";

const NotFound = () => {
  return (
    <>
      <Container>
        <Center>
          <Heading as="h1">404</Heading>
        </Center>
        <Center>
          <Text>The page you&apos;re looking for was not found.</Text>
        </Center>
        <Box my={6} display="flex" alignItems="center" justifyContent="center">
          <NextLink href="/" passHref>
            <Button>
              Return to homepage
            </Button>
          </NextLink>
        </Box>
      </Container>
    </>
  );
};

export default NotFound;
